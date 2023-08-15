#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: lukefanguna
"""
import os
import argparse
import tempfile
import numpy as np
from datetime import datetime, timedelta
from scipy.io import loadmat, savemat
from scipy import signal
from functions import data2index, empdis
from api import VPD, EVI, SSM
from utils import stack_raster_months
import pickle
from typing import List, Dict, Any

FDEO_DIR = os.path.dirname(os.path.dirname(__file__))


class FDEO:
    MODEL_PATH = os.path.join(FDEO_DIR, 'data', 'model_coefficientes.pkl')

    def __init__(self):
        # % ID for each Land cover type
        # % 1: Lake
        # % 2: Developed / Urban
        # % 3: Barren Land
        # % 4: Deciduous
        # % 5: Evergreen
        # % 6: Mixed Forest
        # % 7: Shrubland
        # % 8: Herbaceous
        # % 9: Planted / Cultivated
        # % 10: Wetland
        # % 15: Ocean
        self.lc1 = loadmat(os.path.join(FDEO_DIR, 'data', 'lc1.mat'))['lc1']
        self.lc_type_indices = [4, 5, 7, 8, 10]

        # soil moisture (sm) data from 2003-2013
        self.ssm_training_data = loadmat(os.path.join(FDEO_DIR, 'data', 'sm_20032013.mat'))['sm_20032013']

        # enhanced vegetation index (EVI) data from 2003-2013
        self.evi_training_data = loadmat(os.path.join(FDEO_DIR, 'data', 'EVI_20032013.mat'))['EVI_20032013']

        # vapor pressure deficit (vpd) data from 2003-2013
        self.vpd_training_data = loadmat(os.path.join(FDEO_DIR, 'data', 'vpd_20032013.mat'))['vpd_20032013']

        # Fire data from 2003-2013
        self.firemon_tot_size = loadmat(os.path.join(FDEO_DIR, 'data', 'firemon_tot_size.mat'))['firemon_tot_size']

        self._climatology = self._calculate_training_data_climatology()

        if not os.path.exists(self.MODEL_PATH):
            self.train_model()
        with open(self.MODEL_PATH, 'rb') as file:
            coefficients = pickle.load(file)
        self.coefficients = np.poly1d(coefficients)

        self._lead = 2

    def _calculate_training_data_climatology(self):
        training_data_dimensions = self.firemon_tot_size.shape
        # split the dim sizes for ease of use later on
        firemon_tot_size_x = training_data_dimensions[0]  # lat
        firemon_tot_size_y = training_data_dimensions[1]  # lon

        firemon_tot_size_climatology = np.empty((firemon_tot_size_x, firemon_tot_size_y, 12))
        for i in range(firemon_tot_size_x):
            for j in range(firemon_tot_size_y):
                for k in range(12):
                    firemon_tot_size_climatology[i][j][k] = np.mean(self.firemon_tot_size[i][j][k::12])

        # spatially smooth fire data using a 3 by 3 filter
        smooth_filter = np.ones((3, 3))

        firemon_tot_size_climatology_smoothed_3 = np.empty(firemon_tot_size_climatology.shape)
        for h in range(firemon_tot_size_climatology.shape[2]):
            firemon_tot_size_climatology_smoothed_3[:, :, h] = signal.convolve2d(
                firemon_tot_size_climatology[:, :, h], smooth_filter, mode='same'
            )

        firemon_tot_size_climatology_smoothed_3 /= smooth_filter.size
        return firemon_tot_size_climatology_smoothed_3

    @staticmethod
    def calculate_land_cover_dict(ssm_data: np.array, evi_data: np.array, vpd_data: np.array):
        # Prepare and define training data sets
        # Deciduous DI
        sc_drought = 1  # month range
        deciduous_best_ba_training = data2index(ssm_data, sc_drought)

        # Shrubland DI
        sc_drought = 1
        shrubland_best_ba_training = data2index(evi_data, sc_drought)

        # Evergreen DI
        sc_drought = 3
        vpd_new_drought = data2index(vpd_data, sc_drought)
        evergreen_best_ba_training = vpd_new_drought

        # Herbaceous DI
        herbaceous_best_ba_training = vpd_new_drought

        # Wetland DI
        sc_drought = 3
        wetland_best_ba_training = data2index(ssm_data, sc_drought)

        # List of land-cover IDs according to below
        # 4: Deciduous
        # 5: Evergreen
        # 7: Shrubland
        # 8: Herbaceous
        # 10: Wetland

        return [
            {
                'data': deciduous_best_ba_training,
                'index': 4
            },
            {
                'data': evergreen_best_ba_training,
                'index': 5
            },
            {
                'data': shrubland_best_ba_training,
                'index': 7
            },
            {
                'data': herbaceous_best_ba_training,
                'index': 8
            },
            {
                'data': wetland_best_ba_training,
                'index': 10
            }
        ]

    def train_model(self):
        # This part of the code creates a regression model for each LC type based on
        # the "best" drought indicator (DI) and then creates a historical record of
        # probabilistic and categorical wildfire prediction and observation data
        land_cover_training_data = self.calculate_land_cover_dict(self.ssm_training_data, self.evi_training_data,
                                                                  self.vpd_training_data)

        firemon_tot_shape = self.firemon_tot_size.shape
        for lc_forecast in land_cover_training_data:
            # First build the regression model for the LC Type initial parameters
            mat = np.empty((2, self.firemon_tot_size.size), dtype=float)  # Initial array
            m = 0  # Burned area for each LC. 1-5 is each diff one
            lead = 2  # Lead time
            for i in range(firemon_tot_shape[0]):
                for j in range(firemon_tot_shape[1]):
                    if self.lc1[i][j] == lc_forecast['index']:
                        for k in range(firemon_tot_shape[2]):
                            # Leave the first 3-month empty to account for 2-month lead model development
                            if k - lead < 0:
                                mat[0, m] = np.nan
                            else:
                                mat[0, m] = lc_forecast['data'][i, j, k - lead]
                                mat[1, m] = self.firemon_tot_size[i, j, k]
                            m += 1

            # Remove NaN values from the data
            # Find indices of NaN values in each column
            idx_nan_1 = np.isnan(mat[0, :])
            idx_nan_2 = np.isnan(mat[1, :])

            # Combine indices of NaN values
            idx_nan = np.logical_or(idx_nan_1, idx_nan_2)

            # Filter the array to remove rows with NaN values
            mat = np.vstack([mat[0, :][~idx_nan], mat[1, :][~idx_nan]])

            # Bar plots to derive the regression model
            # Define number of bins
            n_bins = 10

            # Derive min and max of DI (drought indicator) data
            min_1 = np.min(mat[0, :])
            max_1 = np.max(mat[0, :])

            # Derive vector of bins based on DI data
            varbin = np.linspace(min_1, max_1, num=n_bins+1)

            # Initialize arrays to store results
            sample_size = np.zeros(n_bins+1)
            fire_freq_ave = np.zeros(n_bins+1)
            prob = np.zeros(n_bins+1)

            # Find observations in each bin
            for i in range(n_bins+1):
                # Find observations in each bin
                idx3 = np.where((mat[0, :] >= varbin[i]) & (mat[0, :] <= (max_1 + ((max_1 - min_1) / n_bins) if
                                                                          i == n_bins else varbin[i+1])))

                # Find DI in each bin
                # Get corresponding burned area in each bin
                fire_freq_range = mat[1, :][idx3]
                # Calculate number of observations in each bin
                sample_size[i] = len(idx3[0])
                # Calculate sum burned area in each bin
                fire_freq_ave[i] = np.sum(fire_freq_range)
                # Calculate probability of fire at each bin
                prob[i] = fire_freq_ave[i] / sample_size[i]

            # Develop linear regression model
            x = varbin
            y = prob

            # Remove NaN from the observation input
            idx_nan = np.isnan(y)
            x = x[~idx_nan]
            y = y[~idx_nan]

            # Fit the regression model
            coefficients = np.polyfit(x, y, 2)

            # Save the coefficients to a file using pickle
            with open(self.MODEL_PATH, 'wb') as file:
                pickle.dump(coefficients, file)

    def _subtract_climatology(self, fire_data, fire_data_start_date: datetime):
        climatology_index = fire_data_start_date.month - 1
        for i in range(0, fire_data.shape[2]):
            fire_data[:, :, i] = fire_data[:, :, i] - self._climatology[:, :, climatology_index]
            climatology_index += 1
            if climatology_index == 12:
                climatology_index = 0

        return fire_data

    def plot_prob_and_categorical(self, fire_data: np.array, fire_data_start_date: datetime, output_prob_file: str,
                                  output_cat_file: str):
        # Subtract prediction and observation from climatology to derive anomalies
        # TODO: For new prediction data, will need to match up the month of the new data to the month index of
        #  climatology
        fire_data = self._subtract_climatology(fire_data, fire_data_start_date)

        val_new_prob = np.zeros_like(fire_data)
        val_new_cat = np.zeros_like(fire_data)
        for k in range(fire_data.shape[2]):
            # matrix of observation and prediction anomalies for each month
            val_new_prob_m = fire_data[:, :, k]
            val_new_cat_m = fire_data[:, :, k]

            # derive CDF for each LC type
            for lc_type in self.lc_type_indices:
                # derive observation and prediction anomalies for each LC Type
                idx_lc = np.where(self.lc1 == lc_type)
                mat = val_new_prob_m[idx_lc]
                mat = mat.reshape((len(mat), 1))

                # probabilistic CDF
                y = empdis(mat).flatten()

                # 33 percentile threshold for observation time series
                T1 = np.min(y)
                T2 = np.max(y)
                T3 = (T2 - T1) / 3
                T4 = T1 + T3
                T5 = y - T4
                T6 = np.abs(T5)
                T7 = np.where(T6 == np.min(T6))
                below_no_obs = mat[T7[0][0]]

                # 66 percentile threshold for observation time series
                T9 = T4 + T3
                T10 = y - T9
                T11 = np.abs(T10)
                T12 = np.where(T11 == np.min(T11))
                above_no_obs = mat[T12[0][0]]

                # populate categorical prediction matrix
                for i in range(fire_data.shape[0]):
                    for j in range(fire_data.shape[1]):
                        if self.lc1[i, j] == lc_type and val_new_cat_m[i, j] < below_no_obs:
                            val_new_cat_m[i, j] = -1
                        elif self.lc1[i, j] == lc_type and val_new_cat_m[i, j] > above_no_obs:
                            val_new_cat_m[i, j] = 1
                        elif self.lc1[i, j] == lc_type and below_no_obs <= val_new_cat_m[i, j] <= above_no_obs:
                            val_new_cat_m[i, j] = 0

                val_new_prob_m[idx_lc] = y
                val_new_cat_m[idx_lc] = y

            # build matrix of CDFs (probabilistic and categorical observation matrices)
            val_new_prob[:, :, k] = val_new_prob_m
            val_new_cat[:, :, k] = val_new_cat_m

        savemat(output_prob_file, {'data': val_new_prob})
        savemat(output_cat_file, {'data': val_new_cat})

    def inference(self, land_cover_types: List[Dict[str, Any]]) -> np.array:
        inference_results_array = np.full(land_cover_types[0]['data'].shape, np.nan)
        for land_cover_type in land_cover_types:
            lc_index = land_cover_type['index']
            data = land_cover_type['data']
            # Now build a historical forecast matrix based on the developed regression model for each LC Type
            for k in range(self._lead, data.shape[2] + self._lead):
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if self.lc1[i, j] == lc_index:
                            inference_results_array[i, j, k] = (
                                    self.coefficients[0] * (data[i, j, k - self._lead] ** 2)
                                    + self.coefficients[1] * data[i, j, k - self._lead]
                                    + self.coefficients[2]
                            )

        return inference_results_array


def main(
       ssm_data: np.array,
       evi_data: np.array,
       vpd_data: np.array,
       data_start_date: datetime
            ) -> None:
    """
    Reads input data and creates smoothed climatology of wildfire burned data. Default training period is 2003-2013.

    Args:
        ssm_data (np.array): Path to the ssm datafile used for new predictions
        vpd_data (np.array): Path to the vpd datafile used for new predictions
        evi_data (np.array): Path to the evi data used for new predictions
    """
    # On initialization the model will be looked for. If it does not exist then the model will be trained and saved for
    # future initializations
    fdeo = FDEO()

    # First see if training data observation results have been made, if not create them
    training_data_obs_prob_file = os.path.join(FDEO_DIR, 'data', 'training_data_results', 'obs_probability.mat')
    training_data_obs_cat_file = os.path.join(FDEO_DIR, 'data', 'training_data_results', 'obs_categorical.mat')
    if not os.path.exists(training_data_obs_prob_file) or not os.path.exists(training_data_obs_cat_file):
        fdeo.plot_prob_and_categorical(fdeo.firemon_tot_size, datetime(2003, 1, 1), training_data_obs_prob_file,
                                       training_data_obs_cat_file)

    # See if training data prediction results have been made, if not create them
    training_data_pred_prob_file = os.path.join(FDEO_DIR, 'data', 'training_data_results', 'pred_probability.txt')
    training_data_pred_cat_file = os.path.join(FDEO_DIR, 'data', 'training_data_results', 'pred_categorical.txt')
    if not os.path.exists(training_data_pred_prob_file) or not os.path.exists(training_data_pred_cat_file):
        training_land_cover_dict = fdeo.calculate_land_cover_dict(fdeo.ssm_training_data, fdeo.evi_training_data,
                                                                  fdeo.vpd_training_data)
        training_data_fire_inference = fdeo.inference(training_land_cover_dict)
        fdeo.plot_prob_and_categorical(training_data_fire_inference, datetime(2003, 1, 1), training_data_pred_prob_file,
                                       training_data_pred_cat_file)

    # Make predictions with new data
    prediction_land_cover_dict = fdeo.calculate_land_cover_dict(ssm_data, evi_data, vpd_data)
    prediction_data_fire_inference = fdeo.inference(prediction_land_cover_dict)
    fdeo.plot_prob_and_categorical(prediction_data_fire_inference, data_start_date, 'pred_prob_test.tif',
                                   'pred_cat_test.tif')


if __name__ == '__main__':
    # TODO: Add option to input existing csv file other than default
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, required=False,
                        help='Begin date in YYYY-MM-DD format of datafiles to be downloaded from API')
    parser.add_argument('--end_date', type=str, required=False,
                        help='End date in YYYY-MM-DD format of datafiles to be downloaded from API')
    parser.add_argument('-u', '--username', type=str, required=False, dest='username',
                        help='Username to https://urs.earthdata.nasa.gov/ . '
                             'Credentials can also be provided by providing a value to --credentials argument')
    parser.add_argument('-p', '--password', type=str, required=False, dest='password',
                        help='Password to https://urs.earthdata.nasa.gov/ . '
                             'Credentials can also be provided by providing a value to --credentials argument')
    parser.add_argument('-c', '--credentials', type=str, required=False, dest='credentials',
                        help='Path to file containing username and then password separated by newline for '
                             'https://urs.earthdata.nasa.gov/ ')

    parser.add_argument('--ssm_test', type=str, required=False,
                        help='Path to file containing username and then password separated by newline for '
                             'https://urs.earthdata.nasa.gov/ ')
    parser.add_argument('--evi_test', type=str, required=False,
                        help='Path to file containing username and then password separated by newline for '
                             'https://urs.earthdata.nasa.gov/ ')
    parser.add_argument('--vpd_test', type=str, required=False,
                        help='Path to file containing username and then password separated by newline for '
                             'https://urs.earthdata.nasa.gov/ ')

    stacked_ssm_data = None
    stacked_vpd_data = None
    stacked_evi_data = None

    args = parser.parse_args()

    # Make predictions for this month and the next month
    if args.start_date is None and args.end_date is None:
        two_months_ago = datetime.now() - timedelta(weeks=8)
        next_month = two_months_ago.replace(day=28) + timedelta(days=4)
        start_date = two_months_ago.replace(day=1)
        end_date = next_month - timedelta(days=next_month.day)

    elif args.start_date is not None and args.end_date is not None:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date is not None else None
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date is not None else None

    else:
        raise ValueError('Must specify both start date and end date or neither')

    username = args.username
    password = args.password
    if args.credentials is not None:
        with open(args.credentials, 'r') as f:
            lines = f.readlines()
            username = lines[0].strip('\n').strip(' ')
            password = lines[1].strip('\n').strip(' ')

    if username is None or password is None:
        raise ValueError('Must supply https://urs.earthdata.nasa.gov/ credentials with --credentials argument'
                         ' or -u and -p arguments if you would like to download from the API')

    # Download all of the data
    ssm = SSM(username=username, password=password)
    evi = EVI(username=username, password=password)
    vpd = VPD(username=username, password=password)

    # Create temporary directories for the files
    tempdir = tempfile.mkdtemp(prefix='fdeo')
    ssm_dir = os.path.join(tempdir, 'ssm')
    evi_dir = os.path.join(tempdir, 'evi')
    vpd_dir = os.path.join(tempdir, 'vpd')
    os.makedirs(ssm_dir)
    os.makedirs(evi_dir)
    os.makedirs(vpd_dir)

    ssm_data = ssm.create_clipped_time_series(ssm_dir, start_date, end_date)
    evi_data = evi.create_clipped_time_series(evi_dir, start_date, end_date)
    vpd_data = vpd.create_clipped_time_series(vpd_dir, start_date, end_date)

    sorted_evi_files = evi.sort_tif_files(evi_dir)
    sorted_vpd_files = vpd.sort_tif_files(vpd_dir)
    sorted_ssm_files = ssm.sort_tif_files(ssm_dir)

    stacked_evi_data = stack_raster_months(sorted_evi_files)
    stacked_vpd_data = stack_raster_months(sorted_vpd_files)
    stacked_ssm_data = stack_raster_months(sorted_ssm_files)

    print(stacked_ssm_data.shape)
    print(stacked_evi_data.shape)
    print(stacked_vpd_data.shape)

    main(
        ssm_data=stacked_ssm_data,
        evi_data=stacked_evi_data,
        vpd_data=stacked_vpd_data,
        data_start_date=start_date
        )

    if stacked_ssm_data is not None:
        print(ssm_dir)
        # shutil.rmtree(ssm_dir)
    if stacked_evi_data is not None:
        print(evi_dir)
        # shutil.rmtree(evi_dir)
    if stacked_vpd_data is not None:
        print(vpd_dir)
        # shutil.rmtree(vpd_dir)
