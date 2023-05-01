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

from osgeo import osr
import ogr
import gdal
import numpy as np
from pyhdf.SD import SD, SDC
import rasterio
from rasterio.mask import mask
import geopandas as gpd


# TODO: Make sure area of interest being requested is only CONUS


class BaseAPI:
    """
    Defines all the attributes and methods common to the child APIs.
    """
    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, username: str = None, password: str = None):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username = username
        self._password = password
        self._core_count = os.cpu_count()
        self._configure()

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

        if'REQUESTS_CA_BUNDLE' not in os.environ or os.environ['REQUESTS_CA_BUNDLE'] != ssl_cert_path:
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
                       gdal_data_type: int = gdal.GDT_UInt16,
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
                               projection, n_band: int = 1, no_data: int = 0, gdal_data_type: int = gdal.GDT_UInt32):
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

    def _clip_to_conus(self, input_array: np.array, output_tif_file: str):
        num_rows = 3600
        num_cols = 7200

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)

        lon_min, lat_max = -180, 90
        lon_max, lat_min = 180, -90

        # Define the resolution of the raster in degrees
        lon_res = (lon_max - lon_min) / num_cols
        lat_res = (lat_max - lat_min) / num_rows

        # Define the geotransform array in lat/lon
        geotransform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]

        print(geotransform)

        tiff_file = self._numpy_array_to_raster(output_tif_file, input_array, geotransform, 'wgs84')

        with rasterio.open(tiff_file) as src:
            with open(os.path.join(self.PROJ_DIR, 'data', 'CONUS_WGS84.geojson')) as f:
                geojson = json.load(f)
            polygon = gpd.GeoDataFrame.from_features(geojson['features'])

            # Extract the data using the polygon to create a mask
            out_image, out_transform = mask(src, polygon.geometry, nodata=0, crop=True)

            # Update the metadata of the output tif file
            out_meta = src.meta.copy()

            # Open the GeoJSON file containing the polygon

            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                             "transform": out_transform, "dtype": 'int32',
                             'scale': 1/10000
                             })
            # Write the clipped tif file to disk
            with rasterio.open(tiff_file.replace('.tif', '_conus.tif'), "w", **out_meta) as dest:
                dest.write(out_image)


class SSM(BaseAPI):
    """
    Defines all the attributes and methods specific to the OPeNDAP API. This API is used to request and download
    soil moisture data from the GRACE mission.
    """
    _BASE_URL = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GRACEDA/GRACEDADM_CLSM0125US_7D.4.0/'

    def __init__(self, username: str = None, password: str = None):
        super().__init__(username=username, password=password)
        self._dates = self._retrieve_dates()

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}'
        links = self.retrieve_links(self._BASE_URL)
        return sorted([datetime.strptime(link.strip('/'), '%Y') for link in links if re.match(date_re, link) is not
                       None])

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
        nc4_re = r'GRACEDADM\_CLSM0125US\_7D\.A(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.040\.nc4$'
        for date in date_range:
            url = urllib.parse.urljoin(self._BASE_URL, date.strftime('%Y') + '/')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(nc4_re, file)

                if match is not None:
                    date_objs = match.groupdict()
                    file_date = datetime(int(date_objs['year']), int(date_objs['month']), int(date_objs['day']))

                    if t_start <= file_date <= t_stop:
                        remote = urllib.parse.urljoin(url, file)
                        dest = os.path.join(outdir, file)
                        req = (remote, dest)
                        if req not in queries:
                            queries.append(req)
        super().download_time_series(queries, outdir)
        return outdir


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
        hdf_re = r'AIRS\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.L3\.RetStd_IR\d{3}\.v\d+\.\d+\.\d+\.\d+\.G\d{11}\.hdf\.html$'
        for date in date_range:
            url = urllib.parse.urljoin(self._BASE_URL, date.strftime('%Y') + '/' + 'contents.html')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(hdf_re, file)

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
        geotransform = [lon_min, lon_res, 0, lat_max, 0, -lat_res]

        print(geotransform)

        tiff_file = self._numpy_array_to_raster(output_tif_file, input_array, geotransform, 'wgs84')

        with rasterio.open(tiff_file) as src:
            with open(os.path.join(self.PROJ_DIR, 'data', 'CONUS_WGS84.geojson')) as f:
                geojson = json.load(f)
            polygon = gpd.GeoDataFrame.from_features(geojson['features'])

            # Extract the data using the polygon to create a mask
            out_image, out_transform = mask(src, polygon.geometry, nodata=0, crop=True)

            # Update the metadata of the output tif file
            out_meta = src.meta.copy()

            # Open the GeoJSON file containing the polygon

            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                             "transform": out_transform, "dtype": 'int32'
                             })
            # Write the clipped tif file to disk
            with rasterio.open(tiff_file.replace('.tif', '_conus.tif'), "w", **out_meta) as dest:
                dest.write(out_image)

    @staticmethod
    def calculate_vpd(vpd_file: str):
        vpd = SD(vpd_file, SDC.READ)
        rel_hum = np.clip(vpd.select('RelHumSurf_A').get(), a_min=0, a_max=None)
        surf_temp = vpd.select('SurfAirTemp_A').get() - 273.15

        a = (17.625 * surf_temp) / (243.04 + surf_temp)
        b = (17.625 - np.log(rel_hum / 100) - (17.625 * surf_temp)) / (243.04 + surf_temp)
        td = 243.04 * (np.log(rel_hum / 100) + a) / b
        vpd = 0.611 * np.exp((17.5 * surf_temp) / (240.978 + surf_temp)) - (
                    0.611 * np.exp((17.5 * td) / (240.978 + td)))

        return vpd

    def create_stacked_time_series(self, output_tiff_file: str, t_start: datetime = None, t_stop: datetime = None):
        time_series_dir = self.download_time_series(t_start, t_stop)

        arrays = []
        for file in os.listdir(time_series_dir):
            vpd_array = self.calculate_vpd(os.path.join(time_series_dir, file))
            arrays.append(vpd_array)

        stacked_arrays = np.vstack(arrays)

        # TODO: Up sample to 0.25 deg resolution

        self._clip_to_conus(stacked_arrays, output_tiff_file)

        shutil.rmtree(time_series_dir)




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
        hdf_re = r'MOD13C2\.A2\d{6}\.006\.\d{13}\.hdf$'
        for date in date_range:
            url = urllib.parse.urljoin(self._BASE_URL, date.strftime('%Y.%m.%d') + '/')
            files = self.retrieve_links(url)
            for file in files:
                if re.match(hdf_re, file) is not None:
                    remote = urllib.parse.urljoin(url, file)
                    dest = os.path.join(self._TEMP_DIR if outdir is None else outdir, file)
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

    @staticmethod
    def _down_sample(input_array: np.array) -> np.array:
        # Calculate the number of rows and columns in the downsampled array
        rows = input_array.shape[0] // 5
        cols = input_array.shape[1] // 5

        # Create an empty array to hold the downsampled values
        output_array = np.zeros((rows, cols))

        # Loop over the down sampled array and populate with the average of the 0.05 x 0.05 degree pixels
        for i in range(rows):
            for j in range(cols):
                output_array[i, j] = np.mean(output_array[5 * i:5 * (i + 1), 5 * j:5 * (j + 1)])

        return output_array

    def create_stacked_time_series(self, output_tiff_file: str, t_start: datetime = None, t_stop: datetime = None):
        time_series_dir = self.download_time_series(t_start, t_stop)

        arrays = []
        for file in os.listdir(time_series_dir):
            hdf_file = SD(os.path.join(time_series_dir, file), SDC.READ)

            dataset = hdf_file.select('CMG 0.05 Deg Monthly EVI')

            arrays.append(np.array(dataset.get()))

        stacked_arrays = np.vstack(arrays)

        down_sampled_array = self._down_sample(stacked_arrays)

        self._clip_to_conus(down_sampled_array, output_tiff_file)

        shutil.rmtree(time_series_dir)
