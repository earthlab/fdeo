import os
import shutil
import tempfile

import certifi
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
import re
import netCDF4
from pyhdf.SD import SD
from fdeo.api import BaseAPI, VPD, SSM, EVI

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


class TestBaseAPI(unittest.TestCase):
    def test_initialize_already_configured(self):
        """
        Verify that the base class can be instantiated and the user's credentials are read in properly
        """
        b = BaseAPI(username='test_user', password='test_pass')
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        self.assertTrue(b._core_count > 0)
        self.assertEqual(certifi.where(), os.environ['SSL_CERT_FILE'])
        self.assertEqual(certifi.where(), os.environ['REQUESTS_CA_BUNDLE'])

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_blank_cred(self, user_mock, pass_mock):
        """
        Verify that if the class is not instantiated with credentials then they are queried for
        """
        b = BaseAPI()
        self.assertEqual('test_user', b._username)
        self.assertEqual('test_pass', b._password)

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    def test_retrieve_links(self):
        """
        Verify the retrieve links method returns the correct data
        """
        b = BaseAPI(username='test_user', password='test_pass')
        links = b.retrieve_links('https://hydro1.gesdisc.eosdis.nasa.gov/data/GRACEDA/GRACEDADM_CLSM0125US_7D.4.0/')
        self.assertEqual(sorted(['https://www.nasa.gov', 'https://disc.gsfc.nasa.gov/data-access',
                                 'https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Download%20Data%20Files'
                                 '%20from%20HTTPS%20Service%20with%20wget',
                                 'https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ',
                                 '/data/GRACEDA/', '/data/GRACEDA/', '2002/', '2002/', '2003/', '2003/', '2004/',
                                 '2004/', '2005/', '2005/', '2006/', '2006/', '2007/', '2007/', '2008/', '2008/',
                                 '2009/', '2009/', '2010/', '2010/', '2011/', '2011/', '2012/', '2012/', '2013/',
                                 '2013/', '2014/', '2014/', '2015/', '2015/', '2016/', '2016/', '2017/', '2017/',
                                 '2018/', '2018/', '2019/', '2019/', '2020/', '2020/', '2021/', '2021/', '2022/',
                                 '2022/', 'GRACEDADM_CLSM0125US_7D.xml', 'GRACEDADM_CLSM0125US_7D.xml',
                                 'GRACEDADM_CLSM0125US_7D_4.0_dif.xml', 'GRACEDADM_CLSM0125US_7D_4.0_dif.xml',
                                 'doc/', 'doc/', 'https://www.nasa.gov/about/highlights/HP_Privacy.html',
                                 'https://disc.gsfc.nasa.gov/contact']), sorted(links))


class TestSSMAPI(unittest.TestCase):
    username = None
    password = None

    @classmethod
    def setUpClass(cls):
        cred_path = os.path.join(PROJ_DIR, 'credentials.txt')
        if not os.path.exists(cred_path):
            print(f'To run the tests please add a file called credentials.txt to {PROJ_DIR} with your NASA EarthData'
                  f'credentials separated by a newline. If you do not have EarthData credentials you can register for'
                  f'them at https://urs.earthdata.nasa.gov/users/new')

        with open(cred_path, 'r') as f:
            lines = f.readlines()
            cls.username = lines[0].strip(' ').strip("\n")
            cls.password = lines[1].strip(' ').strip("\n")

        return cls

    def setUp(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def tearDown(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def test_initialize_already_configured(self):
        """
        Verify that the base class can be instantiated and the user's credentials are read in properly
        """
        b = SSM(username='test_pass', password='test_pass')
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        self.assertTrue(b._core_count > 0)
        self.assertEqual(certifi.where(), os.environ['SSL_CERT_FILE'])
        self.assertEqual(certifi.where(), os.environ['REQUESTS_CA_BUNDLE'])

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_blank_cred(self, user_mock, pass_mock):
        """
        Verify that if the class is not instantiated with credentials then they are queried for
        """
        b = SSM()
        self.assertEqual('test_user', b._username)
        self.assertEqual('test_pass', b._password)

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    def test_download_time_series(self):
        """
        After successful configuration, verify that valid files can be downloaded for a range of time.
        """
        b = SSM(username=self.username, password=self.password)
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        start_date = datetime(2005, 1, 1)
        end_date = datetime(2005, 1, 20)
        tmp_dir = b.download_time_series(start_date, end_date)
        self.assertTrue(os.path.exists(tmp_dir))
        self.assertTrue(os.listdir(tmp_dir))

        # Make sure they are valid files and within the time range
        for file in os.listdir(tmp_dir):
            t = netCDF4.Dataset(os.path.join(tmp_dir, file))
            self.assertTrue(start_date <= datetime.strptime(t['time'].begin_date, '%Y%M%d') <= end_date)
            self.assertEqual('GRACE Data Assimilation Drought Indicator', t.title)
            self.assertEqual('Catchment', t.source)
            self.assertEqual(sorted(['lat', 'lon', 'time', 'gws_inst', 'rtzsm_inst', 'sfsm_inst']),
                             sorted(list(t.variables.keys())))

            for var in list(t.variables.keys()):
                self.assertTrue(len(t[var][:]) > 0)


class TestVPDAPI(unittest.TestCase):
    username = None
    password = None

    @classmethod
    def setUpClass(cls):
        cred_path = os.path.join(PROJ_DIR, 'credentials.txt')

        if not os.path.exists(cred_path):
            print(f'To run the tests please add a file called credentials.txt to {PROJ_DIR} with your NASA EarthData'
                  f'credentials separated by a newline. If you do not have EarthData credentials you can register for'
                  f'them at https://urs.earthdata.nasa.gov/users/new')

        with open(cred_path, 'r') as f:
            lines = f.readlines()
            cls.username = lines[0].strip(' ').strip("\n")
            cls.password = lines[1].strip(' ').strip("\n")

        return cls

    def setUp(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def tearDown(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def test_initialize_already_configured(self):
        """
        Verify that the base class can be instantiated and the user's credentials are read in properly
        """
        b = VPD(username='test_user', password='test_pass')
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        self.assertTrue(b._core_count > 0)
        self.assertEqual(certifi.where(), os.environ['SSL_CERT_FILE'])
        self.assertEqual(certifi.where(), os.environ['REQUESTS_CA_BUNDLE'])

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_blank_cred(self, user_mock, pass_mock):
        """
        Verify that if the class is not instantiated with credentials then they are queried for
        """
        b = VPD()
        self.assertEqual('test_user', b._username)
        self.assertEqual('test_pass', b._password)

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    def test_download_time_series(self):
        """
        After successful configuration, verify that valid files can be downloaded for a range of time.
        """
        b = VPD(username=self.username, password=self.password)
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        start_date = datetime(2005, 1, 1)
        end_date = datetime(2005, 1, 30)
        tmp_dir = b.download_time_series(start_date, end_date)
        self.assertTrue(os.path.exists(tmp_dir))
        self.assertTrue(os.listdir(tmp_dir))

        # Make sure they are valid files and within the time range
        hdf_re = r'AIRS\.(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})\.L3\.RetSup_IR0\d{2}\.v6\.0\.9\.0\.G\d{11}\.hdf$'
        for file in os.listdir(tmp_dir):
            t = SD(os.path.join(tmp_dir, file))
            match = re.match(hdf_re, file)
            self.assertTrue(match is not None)
            g = match.groupdict()
            file_date = datetime(int(g['year']), int(g['month']), int(g['day']))
            self.assertTrue(start_date <= file_date <= end_date)
            self.assertTrue(len(t.datasets().keys()) > 0)
            for var in list(t.datasets().keys()):
                self.assertTrue(len(t.select(var).get()) > 0)


class TestEVIAPI(unittest.TestCase):
    username = None
    password = None

    @classmethod
    def setUpClass(cls):
        cred_path = os.path.join(PROJ_DIR, 'credentials.txt')

        if not os.path.exists(cred_path):
            print(f'To run the tests please add a file called credentials.txt to {PROJ_DIR} with your NASA EarthData'
                  f'credentials separated by a newline. If you do not have EarthData credentials you can register for'
                  f'them at https://urs.earthdata.nasa.gov/users/new')

        with open(cred_path, 'r') as f:
            lines = f.readlines()
            cls.username = lines[0].strip(' ').strip("\n")
            cls.password = lines[1].strip(' ').strip("\n")

        return cls

    def setUp(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def tearDown(self) -> None:
        temp_path = tempfile.gettempdir()
        for directory in os.listdir(temp_path):
            if directory.startswith('fdeo') and os.path.isdir(directory):
                shutil.rmtree(os.path.join(temp_path, directory))

    def test_initialize_already_configured(self):
        """
        Verify that the base class can be instantiated and the user's credentials are read in properly
        """
        b = EVI(username='test_user', password='test_pass')
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        self.assertTrue(b._core_count > 0)
        self.assertEqual(certifi.where(), os.environ['SSL_CERT_FILE'])
        self.assertEqual(certifi.where(), os.environ['REQUESTS_CA_BUNDLE'])

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_blank_cred(self, user_mock, pass_mock):
        """
        Verify that if the class is not instantiated with credentials then they are queried for
        """
        b = EVI()
        self.assertEqual('test_user', b._username)
        self.assertEqual('test_pass', b._password)

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    def test_download_time_series(self):
        """
        After successful configuration, verify that valid files can be downloaded for a range of time.
        """
        b = EVI(username=self.username, password=self.password)
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        start_date = datetime(2005, 1, 1)
        end_date = datetime(2005, 1, 30)
        tmp_dir = b.download_time_series(start_date, end_date)
        self.assertTrue(os.path.exists(tmp_dir))
        self.assertTrue(os.listdir(tmp_dir))

        # Make sure they are valid files and within the time range
        hdf_re = r'MOD13C2\.A2\d{6}\.006\.\d{13}\.hdf$'
        for file in os.listdir(tmp_dir):
            t = SD(os.path.join(tmp_dir, file))
            d = t.attributes()['ArchiveMetadata.0'].split('DAYSPROCESSED')[1].split("(")[1].split(")")[0].split(",")
            dates = [datetime.strptime(v.strip('"').strip(' "').strip('\n').strip(' '), '%Y%j') for v in d]
            self.assertTrue(all(start_date <= date <= end_date + timedelta(days=31) for date in dates))
            match = re.match(hdf_re, file)
            self.assertTrue(match is not None)


if __name__ == '__main__':
    unittest.main()
