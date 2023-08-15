import getpass
import os
import re
import shutil
import sys
import urllib
import tempfile
from datetime import datetime
from http.cookiejar import CookieJar
from multiprocessing import Pool
from typing import Tuple, List

import json
import certifi
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from osgeo import osr, ogr, gdal
import numpy as np
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import netCDF4 as nc
from fdeo.utils import set_tiff_resolution


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    LAND_COVER_GEOTRANSFORM = [-126.75, 0.25, 0, 23.875, 0, 0.25]
    LAND_COVER_X_SIZE = 244
    LAND_COVER_Y_SIZE = 112

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username = username
        self._password = password
        self._core_count = os.cpu_count()
        if not lazy:
            self._configure()
        self._file_re = None
        self._tif_re = None

    @staticmethod
    def retrieve_links(url: str) -> List[str]:
        """
        Creates a list of all the links found on a webpage
        Args:
            url (str): The URL of the webpage for which you would like a list of links

        Returns:
            (list): All the links on the input URL's webpage
        """
        request = requests.get(url)
        soup = BeautifulSoup(request.text, 'html.parser')
        return [link.get('href') for link in soup.find_all('a')]

    @staticmethod
    def _cred_query() -> Tuple[str, str]:
        """
        Ask the user for their urs.earthdata.nasa.gov username and login
        Returns:
            username (str): urs.earthdata.nasa.gov username
            password (str): urs.earthdata.nasa.gov password
        """
        print('Please input your earthdata.nasa.gov username and password. If you do not have one, you can register'
              ' here: https://urs.earthdata.nasa.gov/users/new')
        username = input('Username:')
        password = getpass.getpass('Password:', stream=None)

        return username, password

    def _configure(self) -> None:
        """
        Queries the user for credentials and configures SSL certificates
        """
        if self._username is None or self._password is None:
            username, password = self._cred_query()

            self._username = username
            self._password = password

        # This is a macOS thing... need to find path to SSL certificates and set the following environment variables
        ssl_cert_path = certifi.where()
        if 'SSL_CERT_FILE' not in os.environ or os.environ['SSL_CERT_FILE'] != ssl_cert_path:
            os.environ['SSL_CERT_FILE'] = ssl_cert_path

        if 'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
            os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path

    def _download(self, query: Tuple[str, str]) -> None:
        """
        Downloads data from the NASA earthdata servers. Authentication is established using the username and password
        found in the local ~/.netrc file.
        Args:
            query (tuple): Contains the remote location and the local path destination, respectively
        """
        link = query[0]
        dest = query[1]

        if os.path.exists(dest):
            return

        pm = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        pm.add_password(None, "https://urs.earthdata.nasa.gov", self._username, self._password)
        cookie_jar = CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPBasicAuthHandler(pm),
            urllib.request.HTTPCookieProcessor(cookie_jar)
        )
        urllib.request.install_opener(opener)
        myrequest = urllib.request.Request(link)
        response = urllib.request.urlopen(myrequest)
        response.begin()
        with open(dest, 'wb') as fd:
            while True:
                chunk = response.read()
                if chunk:
                    fd.write(chunk)
                else:
                    break

    def download_time_series(self, queries: List[Tuple[str, str]], outdir: str):
        """
        Attempts to create download requests for each query, if that fails then makes each request in series.
        Args:
            queries (list): List of tuples containing the remote and local locations for each request
        Returns:
            outdir (str): Path to the output file directory
        """
        # From earthlab firedpy package
        if len(queries) > 0:
            print("Retrieving data...")
            try:
                with Pool(int(self._core_count / 2)) as pool:
                    for _ in tqdm(pool.imap_unordered(self._download, queries), total=len(queries)):
                        pass

            except Exception as pe:
                try:
                    _ = [self._download(q) for q in tqdm(queries, position=0, file=sys.stdout)]
                except Exception as e:
                    template = "Download failed: error type {0}:\n{1!r}"
                    message = template.format(type(e).__name__, e.args)
                    print(message)

        print(f'Wrote {len(queries)} files to {outdir}')

    @staticmethod
    def _create_raster(output_path: str, columns: int, rows: int, n_band: int = 1,
                       gdal_data_type: int = gdal.GDT_Float32,
                       driver: str = r'GTiff'):
        """
        Credit:
        https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-
        specifying-nodata-va

        Creates a blank raster for data to be written to
        Args:
            output_path (str): Path where the output tif file will be written to
            columns (int): Number of columns in raster
            rows (int): Number of rows in raster
            n_band (int): Number of bands in raster
            gdal_data_type (int): Data type for data written to raster
            driver (str): Driver for conversion
        """
        # create driver
        driver = gdal.GetDriverByName(driver)

        output_raster = driver.Create(output_path, columns, rows, n_band, eType=gdal_data_type)
        return output_raster

    def _numpy_array_to_raster(self, output_path: str, numpy_array: np.array, geo_transform,
                               projection, n_band: int = 1, no_data: int = 0, gdal_data_type: int = gdal.GDT_Float32):
        """
        Returns a gdal raster data source
        Args:
            output_path (str): Full path to the raster to be written to disk
            numpy_array (np.array): Numpy array containing data to write to raster
            geo_transform (gdal GeoTransform): tuple of six values that represent the top left corner coordinates, the
            pixel size in x and y directions, and the rotation of the image
            n_band (int): The band to write to in the output raster
            no_data (int): Value in numpy array that should be treated as no data
            gdal_data_type (int): Gdal data type of raster (see gdal documentation for list of values)
        """
        rows, columns = numpy_array.shape

        # create output raster
        output_raster = self._create_raster(output_path, int(columns), int(rows), n_band, gdal_data_type)

        output_raster.SetProjection(projection)
        output_raster.SetGeoTransform(geo_transform)
        output_band = output_raster.GetRasterBand(1)
        output_band.SetNoDataValue(no_data)
        output_band.WriteArray(numpy_array)
        output_band.FlushCache()
        output_band.ComputeStatistics(False)

        if not os.path.exists(output_path):
            raise Exception('Failed to create raster: %s' % output_path)

        return output_path

    def sort_tif_files(self, input_dir: str):
        group_dicts = []
        for file in os.listdir(input_dir):
            match = re.match(self._tif_re, file)
            if match:
                group_dicts.append((os.path.join(input_dir, file), match.groupdict()))

        return [w[0] for w in sorted([v for v in group_dicts], key=lambda x: datetime(int(x[1]['year']),
                                                                                      int(x[1]['month']),
                                                                                      int(x[1]['day'])).timestamp())]


class SSM(BaseAPI):
    """
    Defines all the attributes and methods specific to the OPeNDAP API. This API is used to request and download
    soil moisture data from the GLDAS mission.
    """
    # TODO: Try Early product link / re if data cannot be found for the requested time range
    _BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM025_DA1_D.2.2/'
    _BASE_EP_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM025_DA1_D_EP.2.2/'

    def __init__(self, username: str = None, password: str = None, lazy: bool = False):
        super().__init__(username=username, password=password, lazy=lazy)
        if not lazy:
            self._dates = self._retrieve_dates(self._BASE_URL)
            self._early_product_dates = self._retrieve_dates(self._BASE_EP_URL)
        self._file_re = r'GLDAS\_CLSM025\_DA1\_D(?:_EP)?.A(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.022\.nc4$'
        self._tif_re = r'GLDAS\_CLSM025\_DA1\_D(?:_EP)?.A(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.022\.tif$'

    def _retrieve_dates(self, url: str) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        year_re = r'\d{4}'
        month_re = r'\d{2}'
        date_re = r'\d{4}/\d{2}'
        years = self.retrieve_links(url)
        links = []
        for year in years:
            if re.match(year_re, year):
                months = self.retrieve_links(os.path.join(url, year))
                for month in months:
                    if re.match(month_re, month):
                        links.append(os.path.join(year, month))

        return sorted(list(set([datetime.strptime(link, '%Y/%m/') for link in links if re.match(date_re, link) is not
                                None])))

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None) -> str:
        """
        Queries the EarthData site for AIRS Level-3 V6 data in the range of input start and stop
        Args:
            t_start (datetime): Start of the query for data. If none, the earliest data set found will be used.
            Data sets are produced monthly.
            t_stop (datetime): Stop of the query for data. If none, the latest data set found will be used.
            Data sets are produced monthly.
            outdir (str): Path to the output directory where the time series will be written to. The default value is
            CWD/tmp/evi
        """
        if outdir is None:
            outdir = tempfile.mkdtemp(prefix='fdeo')
        else:
            os.makedirs(outdir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._early_product_dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates + self._early_product_dates if
                      t_start.year <= date.year <= t_stop.year]
        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []

        for date in date_range:
            base_url = self._BASE_URL if date in self._dates else self._BASE_EP_URL
            url = urllib.parse.urljoin(base_url, date.strftime('%Y') + '/' + date.strftime('%m') + '/')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(self._file_re, file)

                if match is not None:
                    date_objs = match.groupdict()
                    file_date = datetime(int(date_objs['year']), int(date_objs['month']), int(date_objs['day']))

                    if t_start <= file_date <= t_stop:
                        remote = urllib.parse.urljoin(url, file)
                        dest = os.path.join(outdir, file)
                        if os.path.exists(dest):
                            continue
                        req = (remote, dest)
                        if req not in queries:
                            queries.append(req)
        super().download_time_series(queries, outdir)
        return outdir

    def create_clipped_time_series(self, output_dir: str, t_start: datetime = None, t_stop: datetime = None,
                                   time_series_dir: str = None):
        if time_series_dir is None:
            time_series_dir = self.download_time_series(t_start, t_stop)

        # Take the mean for each day of the month
        files_by_month = {}
        for file in os.listdir(time_series_dir):
            file_path = os.path.join(time_series_dir, file)
            match = re.match(self._file_re, file)

            if match is not None:
                date_objs = match.groupdict()
                date_hash = date_objs['year'] + date_objs['month']
                if date_hash not in files_by_month:
                    files_by_month[date_hash] = [file_path]
                else:
                    files_by_month[date_hash].append(file_path)

        for month_group in files_by_month:
            daily_ssm = []
            files = files_by_month[month_group]
            for file in files:
                try:
                    nc_file = nc.Dataset(file, 'r')
                    dataset_name = 'SoilMoist_S_tavg'
                    daily_ssm.append(nc_file.variables[dataset_name][:])
                except OSError:
                    print(file)
                    continue
            stacked_array = np.stack(daily_ssm, axis=0)
            mean_array = np.mean(stacked_array, axis=0)

            output_tiff_file = os.path.join(output_dir, os.path.basename(files[0]).replace('.nc4', '.tif'))

            self._clip_to_conus(mean_array[0], output_tiff_file)

    def _clip_to_conus(self, input_array: np.array, output_tif_file: str):
        num_rows = 600
        num_cols = 1440

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        lon_min, lat_max = -180, 90
        lon_max, lat_min = 180, -60

        # Define the resolution of the raster in degrees
        lon_res = (lon_max - lon_min) / num_cols
        lat_res = (lat_max - lat_min) / num_rows

        # Define the geotransform array in lat/lon
        input_geotransform = [lon_min, lon_res, 0, lat_min, 0, lat_res]

        # Interpolate input data and then sample each point in land cover
        fixed_to_land_cover = set_tiff_resolution(input_array, input_geotransform, num_cols, num_rows,
                                                  self.LAND_COVER_GEOTRANSFORM, self.LAND_COVER_X_SIZE,
                                                  self.LAND_COVER_Y_SIZE)

        fixed_to_land_cover = np.flip(fixed_to_land_cover, axis=0)
        _ = self._numpy_array_to_raster(output_tif_file, fixed_to_land_cover, self.LAND_COVER_GEOTRANSFORM, 'wgs84',
                                        gdal_data_type=gdal.GDT_Float32)


class VPD(BaseAPI):
    """
    Defines all the attributes and methods specific to the OPeNDAP API. This API is used to request and download
    Level-3 V6 VPD data from the AIRS satellite.
    """
    _BASE_URL = 'https://acdisc.gesdisc.eosdis.nasa.gov/opendap/Aqua_AIRS_Level3/AIRS3STM.7.0/'

    def __init__(self, username: str = None, password: str = None):
        """
        Defines the base URL for Level-3 V6 VPD data and the temporary directory where the requested files will be
        written. Also finds the dates available for request from the server.
        """
        super().__init__(username=username, password=password)
        self._dates = self._retrieve_dates()
        self._file_re = r'AIRS\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.L3\.RetStd_IR\d{3}\.v\d+\.\d+\.\d+\.\d+\.G\d{11}\.hdf\.html$'
        self._tif_re = r'AIRS\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.L3\.RetStd_IR\d{3}\.v\d+\.\d+\.\d+\.\d+\.G\d{11}\.tif$'

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}/contents.html'
        links = self.retrieve_links(self._BASE_URL + 'contents.html')
        return sorted([datetime.strptime(link.strip('/contents.html'), '%Y') for link in links if
                       re.match(date_re, link) is not None])

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None) -> str:
        """
        Queries the EarthData site for AIRS Level-3 V6 data in the range of input start and stop
        Args:
            t_start (datetime): Start of the query for data. If none, the earliest data set found will be used.
            Data sets are produced monthly.
            t_stop (datetime): Stop of the query for data. If none, the latest data set found will be used.
            Data sets are produced monthly.
            outdir (str): Path to the output directory where the time series will be written to. The default value is
            CWD/tmp/evi
        """
        if outdir is None:
            outdir = tempfile.mkdtemp(prefix='fdeo')
        else:
            os.makedirs(outdir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates if t_start.year <= date.year <= t_stop.year]
        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []
        for date in date_range:
            url = urllib.parse.urljoin(self._BASE_URL, date.strftime('%Y') + '/' + 'contents.html')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(self._file_re, file)
                if match is not None:
                    date_objs = match.groupdict()
                    file_date = datetime(int(date_objs['year']), int(date_objs['month']), int(date_objs['day']))

                    if t_start <= file_date <= t_stop:
                        remote = urllib.parse.urljoin(url, file.strip('.html')).replace('opendap', 'data')
                        dest = os.path.join(outdir, file.strip('.html'))
                        req = (remote, dest)
                        if req not in queries:
                            queries.append(req)

        super().download_time_series(queries, outdir)
        return outdir

    def _clip_to_conus(self, input_array: np.array, output_tif_file: str):
        num_rows = 180
        num_cols = 360

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        lon_min, lat_max = -180, 90
        lon_max, lat_min = 180, -90

        # Define the resolution of the raster in degrees
        lon_res = (lon_max - lon_min) / num_cols
        lat_res = (lat_max - lat_min) / num_rows

        # Define the geotransform array in lat/lon
        input_geotransform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]

        # Interpolate input data and then sample each point in land cover
        fixed_to_land_cover = set_tiff_resolution(input_array, input_geotransform, num_cols, num_rows,
                                                  self.LAND_COVER_GEOTRANSFORM, self.LAND_COVER_X_SIZE,
                                                  self.LAND_COVER_Y_SIZE)

        fixed_to_land_cover = np.flip(fixed_to_land_cover, axis=0)
        _ = self._numpy_array_to_raster(output_tif_file, fixed_to_land_cover, self.LAND_COVER_GEOTRANSFORM, 'wgs84',
                                        gdal_data_type=gdal.GDT_Float32)

    @staticmethod
    def calculate_vpd(vpd_file: str):
        vpd = SD(vpd_file, SDC.READ)
        rel_hum = np.clip(vpd.select('RelHumSurf_A').get(), a_min=0, a_max=None)
        surf_temp = vpd.select('SurfAirTemp_A').get() - 273.15

        es = 0.611 * np.exp((17.27 * surf_temp) / (surf_temp + 237.3))  # saturation vapor pressure
        e = (rel_hum / 100) * es  # vapor pressure
        vpd = es - e

        return vpd

    def create_clipped_time_series(self, output_dir: str, t_start: datetime = None, t_stop: datetime = None):
        time_series_dir = self.download_time_series(t_start, t_stop)

        for file in os.listdir(time_series_dir):
            vpd_array = self.calculate_vpd(os.path.join(time_series_dir, file))

            # TODO: Add constants for scale factors and use them in main function
            vpd_array = vpd_array

            output_tiff_file = os.path.join(output_dir, file.replace('.hdf', '.tif'))

            self._clip_to_conus(vpd_array, output_tiff_file)


class EVI(BaseAPI):
    """
    Defines all the attributes and methods specific to the MODIS API. This API is used to request and download
    Enhanced Vegetation Index (EVI) data from the MODIS satellite.
    """
    _BASE_URL = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD13C2.006/'

    def __init__(self, username: str = None, password: str = None):
        """
        Defines the base URL for Enhanced Vegetation Index (EVI) data and the temporary directory where the requested
        files will be written. Also finds the dates available for request from the server.
        """
        super().__init__(username=username, password=password)
        self._dates = self._retrieve_dates()
        self._file_re = r'MOD13C2\.A2\d{6}\.006\.\d{13}\.hdf$'
        self._tif_re = r'MOD13C2\.A2\d{6}\.006\.\d{13}\_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\_\.tif$'

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None) -> str:
        """
        Queries the EarthData site for MODIS Enhanced Vegetation Index (EVI) data in the range of input start and stop
        Args:
            t_start (datetime): Start of the query for data. If none, the earliest data set found will be used.
            Data sets are produced monthly.
            t_stop (datetime): Stop of the query for data. If none, the latest data set found will be used.
            Data sets are produced monthly.
            outdir (str): Path to the output directory where the time series will be written to. The default value is
            CWD/tmp/evi
        """
        if outdir is None:
            outdir = tempfile.mkdtemp(prefix='fdeo')
        else:
            os.makedirs(outdir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates if t_start <= date <= t_stop]

        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []
        for date in date_range:
            url = urllib.parse.urljoin(self._BASE_URL, date.strftime('%Y.%m.%d') + '/')
            files = self.retrieve_links(url)
            for file in files:
                if re.match(self._file_re, file) is not None:
                    remote = urllib.parse.urljoin(url, file)
                    out_file = file.replace('.hdf', f"_{date.strftime('%Y%m%d')}_.hdf")
                    dest = os.path.join(self._TEMP_DIR if outdir is None else outdir, out_file)
                    queries.append((remote, dest))

        super().download_time_series(queries, outdir)
        return outdir

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}\.\d{2}\.\d{2}'
        links = self.retrieve_links(self._BASE_URL)
        return sorted([datetime.strptime(link.strip('/'), '%Y.%m.%d') for link in links if
                       re.match(date_re, link.strip('/')) is not None])

    def create_clipped_time_series(self, output_dir: str, t_start: datetime = None, t_stop: datetime = None):
        time_series_dir = self.download_time_series(t_start, t_stop)

        for file in os.listdir(time_series_dir):
            hdf_file = SD(os.path.join(time_series_dir, file), SDC.READ)

            dataset = hdf_file.select('CMG 0.05 Deg Monthly EVI')

            output_tiff_file = os.path.join(output_dir, file.replace('.hdf', '.tif'))

            arr = dataset.get()

            # Set all fill values to 0
            fill_values = np.where(arr == -3000)
            arr[fill_values] = 0

            # Scale data
            arr = arr * 0.0001

            self._clip_to_conus(arr, output_tiff_file)

    def _clip_to_conus(self, input_array: np.array, output_tif_file: str):
        num_rows = 3600
        num_cols = 7200

        lon_min, lat_max = -180, 90
        lon_max, lat_min = 180, -90

        # Define the resolution of the raster in degrees
        lon_res = (lon_max - lon_min) / num_cols
        lat_res = (lat_max - lat_min) / num_rows

        # Define the geotransform array in lat/lon
        input_geotransform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]

        # Interpolate input data and then sample each point in land cover
        fixed_to_land_cover = set_tiff_resolution(input_array, input_geotransform, num_cols, num_rows,
                                                  self.LAND_COVER_GEOTRANSFORM, self.LAND_COVER_X_SIZE,
                                                  self.LAND_COVER_Y_SIZE)

        fixed_to_land_cover = np.flip(fixed_to_land_cover, axis=0)
        _ = self._numpy_array_to_raster(output_tif_file, fixed_to_land_cover, self.LAND_COVER_GEOTRANSFORM, 'wgs84')
