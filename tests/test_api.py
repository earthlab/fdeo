import os
import certifi
import unittest
from unittest import mock
from unittest.mock import patch
from pathlib import Path
from fdeo.api import BaseAPI, VPD, SSM, EVI


class TestBaseAPI(unittest.TestCase):

    PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
    TEST_NETRC_PATH = os.path.join(PROJ_DIR, 'tests', '.netrc')

    def setUp(self) -> None:
        if os.path.exists(self.TEST_NETRC_PATH):
            os.remove(self.TEST_NETRC_PATH)

    def tearDown(self) -> None:
        if os.path.exists(self.TEST_NETRC_PATH):
            os.remove(self.TEST_NETRC_PATH)

    def test_initialize_already_configured(self):
        """
        Verify that the base class can be instantiated and the user's credentials are read in properly
        """
        b = BaseAPI()
        self.assertIsNotNone(b._username)
        self.assertIsNotNone(b._password)
        self.assertEqual(os.path.join(self.PROJ_DIR, 'data', 'tmp'), b._temp_dir)
        self.assertTrue(b._core_count > 0)
        self.assertEqual(os.path.join(str(Path.home()), '.netrc'), b._NETRC_PATH)
        self.assertEqual(certifi.where(), os.environ['SSL_CERT_FILE'])
        self.assertEqual(certifi.where(), os.environ['REQUESTS_CA_BUNDLE'])

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_no_netrc_file(self, user_mock, pass_mock):
        """
        Verify that if there exists no .netrc file at the netrc_path that one is made and the user is queried for their
        credentials to fill in the file
        """
        self.assertFalse(os.path.exists(self.TEST_NETRC_PATH))

        with mock.patch('fdeo.api.BaseAPI._NETRC_PATH', new_callable=mock.PropertyMock) as mb:
            mb.return_value = self.TEST_NETRC_PATH
            b = BaseAPI()
            self.assertTrue(os.path.exists(self.TEST_NETRC_PATH))
            self.assertEqual('test_user', b._username)
            self.assertEqual('test_pass', b._password)
            with open(self.TEST_NETRC_PATH, 'r') as f:
                lines = f.readlines()
                self.assertEqual(1, len(lines))
                self.assertEqual('machine urs.earthdata.nasa.gov login test_user password test_pass', lines[0])

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_invalid_netrc_file(self, user_mock, pass_mock):
        """
        Verify that if the .netrc file is found but there are no valid earthdata credentials, the user is queried for
        credentials and these are written to the .netrc file
        """
        with open(self.TEST_NETRC_PATH, 'w+') as f:
            f.write('testmachine foo.nasa.gov login foo password foo')

        with mock.patch('fdeo.api.BaseAPI._NETRC_PATH', new_callable=mock.PropertyMock) as mb:
            mb.return_value = self.TEST_NETRC_PATH
            b = BaseAPI()
            self.assertEqual('test_user', b._username)
            self.assertEqual('test_pass', b._password)
            with open(self.TEST_NETRC_PATH, 'r') as f:
                lines = f.readlines()
                self.assertEqual(2, len(lines))
                self.assertEqual('testmachine foo.nasa.gov login foo password foo\n', lines[0])
                self.assertEqual('machine urs.earthdata.nasa.gov login test_user password test_pass', lines[1])

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)

    @patch('builtins.input', return_value='test_user')
    @patch('fdeo.api.getpass.getpass', return_value='test_pass')
    def test_initialize_blank_netrc_file(self, user_mock, pass_mock):
        """
        Verify that if the .netrc file is found but there are no valid earthdata credentials, the user is queried for
        credentials and these are written to the .netrc file
        """
        with open(self.TEST_NETRC_PATH, 'w+') as f:
            f.write('')

        with mock.patch('fdeo.api.BaseAPI._NETRC_PATH', new_callable=mock.PropertyMock) as mb:
            mb.return_value = self.TEST_NETRC_PATH
            b = BaseAPI()
            self.assertEqual('test_user', b._username)
            self.assertEqual('test_pass', b._password)
            with open(self.TEST_NETRC_PATH, 'r') as f:
                lines = f.readlines()
                self.assertEqual(1, len(lines))
                self.assertEqual('machine urs.earthdata.nasa.gov login test_user password test_pass', lines[0])

        self.assertEqual(1, user_mock.call_count)
        self.assertEqual(1, pass_mock.call_count)


if __name__ == '__main__':
    unittest.main()
