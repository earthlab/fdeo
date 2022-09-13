import os
from pathlib import Path
import getpass
from typing import Tuple, List
import re
import urllib
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import sys
import stat
from http.cookiejar import CookieJar
import certifi
import requests


class BaseAPI:
    """
    Defines all of the attributes and methods common to the child APIs.
    """

    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    _NETRC_PATH = os.path.join(str(Path.home()), '.netrc')

    def __init__(self):
        """
        Initializes the common attributes required for each data type's API
        """
        self._username = None
        self._password = None
        self._temp_dir = os.path.join(self.PROJ_DIR, 'data', 'tmp')
        self._core_count = os.cpu_count()
        self._configure()

    @staticmethod
    def retrieve_links(url: str) -> List[str]:
        """
        Creates a list of all of the links found on a webpage
        Args:
            url (str): The URL of the webpage for which you would like a list of links

        Returns:
            (list): All of the links on the input URL's webpage
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
        Checks for the existence of the ~/.netrc file, which should contain the users urs.earthdata.nasa.gov
        credentials. If this file does not exist or the credentials are missing / misformatted, query the user for
        their credentials and write them to the file.
        """
        if not os.path.exists(self._NETRC_PATH):
            username, password = self._cred_query()
            with open(self._NETRC_PATH, 'w+') as f:
                f.write(f'machine urs.earthdata.nasa.gov login {username} password {password}')
            self._username = username
            self._password = password

        else:
            earth_data_re = r'machine urs.earthdata.nasa.gov login (?P<username>.+) password (?P<password>.+)'
            cred_found = False
            with open(self._NETRC_PATH, 'r') as f:
                for line in f.readlines():
                    match = re.match(earth_data_re, line)
                    if match is not None:
                        params = match.groupdict()
                        print(f'urs.earthdata.nasa.gov credentials found in {self._NETRC_PATH}')
                        self._username = params['username']
                        self._password = params['password']
                        cred_found = True
                        break

            if not cred_found:
                print(f'Could not find valid urs.earthdata.nasa.gov credentials at {self._NETRC_PATH} . '
                      f'Credentials will be written to this file once supplied.')
                username, password = self._cred_query()

                with open(self._NETRC_PATH, 'r') as f:
                    text = f.read()

                with open(self._NETRC_PATH, 'a') as f:
                    if not text.endswith('\n') and os.stat(self._NETRC_PATH).st_size != 0:
                        f.write('\n')
                    f.write(f'machine urs.earthdata.nasa.gov login {username} password {password}')
                self._username = username
                self._password = password

        os.chmod(self._NETRC_PATH, stat.S_IWRITE | stat.S_IREAD)

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
            outdir (str)

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

        print(f'Wrote {len(queries)} files to {self._temp_dir if outdir is None else outdir}')


class SSM(BaseAPI):
    """

    """
    def __init__(self):
        super().__init__()
        self._base_url = 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GRACEDA/GRACEDADM_CLSM0125US_7D.4.0/'
        self._temp_dir = os.path.join(self._temp_dir, 'ssm')
        self._dates = self._retrieve_dates()

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}'
        links = self.retrieve_links(self._base_url)
        return sorted([datetime.strptime(link.strip('/'), '%Y') for link in links if re.match(date_re, link) is not
                       None])

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None):
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
        os.makedirs(self._temp_dir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates if t_start.year <= date.year <= t_stop.year]

        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []
        nc4_re = r'GRACEDADM\_CLSM0125US\_7D\.A(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.040\.nc4$'
        for date in date_range:
            url = urllib.parse.urljoin(self._base_url, date.strftime('%Y') + '/')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(nc4_re, file)

                if match is not None:
                    date_objs = match.groupdict()
                    file_date = datetime(int(date_objs['year']), int(date_objs['month']), int(date_objs['day']))

                    if t_start <= file_date <= t_stop:
                        remote = urllib.parse.urljoin(url, file)
                        dest = os.path.join(self._temp_dir if outdir is None else outdir, file)
                        req = (remote, dest)
                        if req not in queries:
                            queries.append(req)

        super().download_time_series(queries, outdir)


class VPD(BaseAPI):
    """
    Defines all of the attributes and methods specific to the OPeNDAP API. This API is used to request and download
    Level-3 V6 VPD data from the AIRS satellite.
    """
    def __init__(self):
        """
        Defines the base URL for Level-3 V6 VPD data and the temporary directory where the requested files will be
        written. Also finds the dates avaialble for request from the server.
        """
        super().__init__()
        self._base_url = 'https://acdisc.gesdisc.eosdis.nasa.gov/opendap/Aqua_AIRS_Level3/AIRS3SPM.006/'
        self._temp_dir = os.path.join(self._temp_dir, 'vpd')
        self._dates = self._retrieve_dates()

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}/contents.html'
        links = self.retrieve_links(self._base_url + 'contents.html')
        return sorted([datetime.strptime(link.strip('/contents.html'), '%Y') for link in links if
                       re.match(date_re, link) is not None])

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None):
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
        os.makedirs(self._temp_dir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates if t_start.year <= date.year <= t_stop.year]

        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []
        hdf_re = r'AIRS\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.L3\.RetSup_IR030\.v6\.0\.9\.0\.G\d{11}\.hdf\.html$'
        for date in date_range:
            url = urllib.parse.urljoin(self._base_url, date.strftime('%Y') + '/' + 'contents.html')
            files = self.retrieve_links(url)

            for file in files:
                match = re.match(hdf_re, file)

                if match is not None:
                    date_objs = match.groupdict()
                    file_date = datetime(int(date_objs['year']), int(date_objs['month']), int(date_objs['day']))

                    if t_start <= file_date <= t_stop:
                        remote = urllib.parse.urljoin(url, file.strip('.html')).replace('opendap', 'data')
                        dest = os.path.join(self._temp_dir if outdir is None else outdir, file.strip('.html'))
                        req = (remote, dest)
                        if req not in queries:
                            queries.append(req)

        super().download_time_series(queries, outdir)


class EVI(BaseAPI):
    """
    Defines all of the attributes and methods specific to the MODIS API. This API is used to request and download
    Enhanced Vegetation Index (EVI) data from the MODIS satellite.
    """

    def __init__(self):
        """
        Defines the base URL for Enhanced Vegetation Index (EVI) data and the temporary directory where the requested
        files will be written. Also finds the dates available for request from the server.
        """
        super().__init__()
        self._base_url = 'https://e4ftl01.cr.usgs.gov/MOLT/MOD13C2.006/'
        self._temp_dir = os.path.join(self._temp_dir, 'evi')
        self._dates = self._retrieve_dates()

    def download_time_series(self, t_start: datetime = None, t_stop: datetime = None, outdir: str = None):
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
        os.makedirs(self._temp_dir, exist_ok=True)

        t_start = self._dates[0] if t_start is None else t_start
        t_stop = self._dates[-1] if t_stop is None else t_stop
        date_range = [date for date in self._dates if t_start <= date <= t_stop]

        if not date_range:
            raise ValueError('There is no data available in the time range requested')

        queries = []
        hdf_re = r'MOD13C2\.A2\d{6}\.006\.\d{13}\.hdf$'
        for date in date_range:
            url = urllib.parse.urljoin(self._base_url, date.strftime('%Y.%m.%d') + '/')
            files = self.retrieve_links(url)
            for file in files:
                if re.match(hdf_re, file) is not None:
                    remote = urllib.parse.urljoin(url, file)
                    dest = os.path.join(self._temp_dir if outdir is None else outdir, file)
                    queries.append((remote, dest))

        super().download_time_series(queries, outdir)

    def _retrieve_dates(self) -> List[datetime]:
        """
        Finds which dates are available from the server and returns them as a list of datetime objects
        Returns:
            (list): List of available dates on the OPeNDAP server in ascending order
        """
        date_re = r'\d{4}\.\d{2}\.\d{2}'
        links = self.retrieve_links(self._base_url)
        return sorted([datetime.strptime(link.strip('/'), '%Y.%m.%d') for link in links if
                       re.match(date_re, link.strip('/')) is not None])
