#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: erickverleye
"""
import os
import argparse
import tempfile
import numpy as np
from datetime import datetime, timedelta
from scipy.io import loadmat, savemat
from scipy import signal
from functions import data2index, empdis
from api import VPD, EVI, SSM, BaseAPI
from utils import stack_raster_months
import pickle
from typing import List, Dict, Any
from osgeo import gdal
import matplotlib.pyplot as plt
import matplotlib as mpl

FDEO_DIR = os.path.dirname(os.path.dirname(__file__))


class FDEO:
    MODEL_PATH = os.path.join(FDEO_DIR, 'data', 'model_coefficients.pkl')
    MONTHS = [
        'January',
        'February',
        'March',
        'April',
        'May',
        'June',
        'July',
        'August',
        'September',
        'October',
        'November',
        'December'
    ]

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
            print(f'Model parameters not found at {self.MODEL_PATH}, training model now')
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

    def _create_plots(self, output_dir: str, results_start_date: datetime, data: np.array, categorical: bool = False):
        n_monts = data.shape[2]
        os.makedirs(output_dir, exist_ok=True)

        for month in range(n_monts):
            results_date = results_start_date + timedelta(weeks=4*month)
            title = f'{self.MONTHS[results_date.month - 1]} {results_date.year}'

            fig, ax = plt.subplots()

            cmaplist = [(0, 128, 0), (255, 255, 255), (128, 0, 0)]
            cmap = plt.cm.jet
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            if categorical:
                norm = mpl.colors.BoundaryNorm([-1, -0.667, 0.667, 1], cmap.N)
                shw = ax.imshow(data[:, :, month], cmap=cmap, norm=norm)
                bar = fig.colorbar(shw, ticks=[])
                bar.ax.text(4, 1, "above normal", fontsize=10, va='center')
                bar.ax.text(4, 0, "normal", fontsize=10, va='center')
                bar.ax.text(4, -1, "below normal", fontsize=10, va='center')
            else:
                shw = ax.imshow(data[:, :, month], cmap=cmap)
                bar = fig.colorbar(shw, ticks=np.arange(0, 1.2, 0.2))
                bar.ax.text(4, 1, "above normal", fontsize=10, va='center')
                bar.ax.text(4, 0.5, "normal", fontsize=10, va='center')
                bar.ax.text(4, 0, "below normal", fontsize=10, va='center')

            # show plot with labels
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            plt.title(title)

            outpath = os.path.join(output_dir, title + '.png')
            plt.savefig(outpath, dpi=300, bbox_inches='tight')

    def calculate_prob_and_categorical(self, fire_data: np.array, fire_data_start_date: datetime, output_prob_file: str,
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
                val_new_prob_m[idx_lc] = y

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

            # build matrix of CDFs (probabilistic and categorical observation matrices)
            val_new_prob[:, :, k] = val_new_prob_m
            val_new_cat[:, :, k] = val_new_cat_m

        BaseAPI._numpy_array_to_raster(output_prob_file, val_new_prob, BaseAPI.LAND_COVER_GEOTRANSFORM, 'wgs84',
                                       n_bands=fire_data.shape[2], gdal_data_type=gdal.GDT_Float32)
        self._create_plots(os.path.join(os.path.dirname(output_prob_file), 'probability_plots'),
                           fire_data_start_date + timedelta(weeks=4*self._lead), val_new_prob)
        BaseAPI._numpy_array_to_raster(output_cat_file, val_new_cat, BaseAPI.LAND_COVER_GEOTRANSFORM, 'wgs84',
                                       n_bands=fire_data.shape[2], gdal_data_type=gdal.GDT_Float32)
        self._create_plots(os.path.join(os.path.dirname(output_cat_file), 'categorical_plots'),
                           fire_data_start_date + timedelta(weeks=4 * self._lead), val_new_cat)

    def inference(self, land_cover_types: List[Dict[str, Any]]) -> np.array:
        inference_results_array = np.full(land_cover_types[0]['data'].shape, np.nan)
        for land_cover_type in land_cover_types:
            lc_index = land_cover_type['index']
            data = land_cover_type['data']
            # Now build a historical forecast matrix based on the developed regression model for each LC Type
            # Each value in the output array is 2 months ahead
            for k in range(data.shape[2]):
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        if self.lc1[i, j] == lc_index:
                            inference_results_array[i, j, k - self._lead] = (
                                    self.coefficients[0] * (data[i, j, k] ** 2)
                                    + self.coefficients[1] * data[i, j, k]
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
    training_data_dir = os.path.join(FDEO_DIR, 'data', 'training_data_results')
    os.makedirs(training_data_dir, exist_ok=True)
    training_data_obs_prob_file = os.path.join(training_data_dir, 'obs_probability.tif')
    training_data_obs_cat_file = os.path.join(training_data_dir, 'obs_categorical.tif')
    if not os.path.exists(training_data_obs_prob_file) or not os.path.exists(training_data_obs_cat_file):
        print(f'Writing training data observation files to {training_data_obs_prob_file}, {training_data_obs_cat_file}')
        fdeo.calculate_prob_and_categorical(fdeo.firemon_tot_size, datetime(2003, 1, 1), training_data_obs_prob_file,
                                            training_data_obs_cat_file)

    # See if training data prediction results have been made, if not create them
    training_data_pred_prob_file = os.path.join(training_data_dir, 'pred_probability.tif')
    training_data_pred_cat_file = os.path.join(training_data_dir, 'pred_categorical.tif')
    if not os.path.exists(training_data_pred_prob_file) or not os.path.exists(training_data_pred_cat_file):
        print(f'Writing training data prediction files to {training_data_pred_prob_file}, {training_data_pred_cat_file}')
        training_land_cover_dict = fdeo.calculate_land_cover_dict(fdeo.ssm_training_data, fdeo.evi_training_data,
                                                                  fdeo.vpd_training_data)
        training_data_fire_inference = fdeo.inference(training_land_cover_dict)
        fdeo.calculate_prob_and_categorical(training_data_fire_inference, datetime(2003, 1, 1),
                                            training_data_pred_prob_file, training_data_pred_cat_file)

    # Make predictions with new data
    print('Plotting predictions for 2 month lead times')
    prediction_land_cover_dict = fdeo.calculate_land_cover_dict(ssm_data, evi_data, vpd_data)
    prediction_data_fire_inference = fdeo.inference(prediction_land_cover_dict)

    # Write output tif files and plots
    results_start_date = start_date + timedelta(weeks=8)
    results_end_date = end_date + timedelta(weeks=8)
    output_dir = os.path.join(FDEO_DIR, 'data', 'prediction_results', f'{results_start_date}_{results_end_date}')
    os.makedirs(output_dir, exist_ok=True)

    fdeo.calculate_prob_and_categorical(prediction_data_fire_inference, data_start_date,
                                        os.path.join(output_dir, 'prediction_probability.tif'),
                                        os.path.join(output_dir, 'predication_categorical.tif'))


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
        start_date = two_months_ago.replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)\
            .replace(microsecond=0)
        end_date = next_month.replace(day=28).replace(hour=23).replace(minute=59).replace(second=59)

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
    data_dir = os.path.join(FDEO_DIR, 'data')
    ssm_dir = os.path.join(data_dir, 'ssm')
    evi_dir = os.path.join(data_dir, 'evi')
    vpd_dir = os.path.join(data_dir, 'vpd')
    os.makedirs(ssm_dir, exist_ok=True)
    os.makedirs(evi_dir, exist_ok=True)
    os.makedirs(vpd_dir, exist_ok=True)

    ssm_data = ssm.create_clipped_time_series(ssm_dir, start_date, end_date)
    evi_data = evi.create_clipped_time_series(evi_dir, start_date, end_date)
    vpd_data = vpd.create_clipped_time_series(vpd_dir, start_date, end_date)

    sorted_evi_files = evi.sort_tif_files(evi_dir)
    sorted_vpd_files = vpd.sort_tif_files(vpd_dir)
    sorted_ssm_files = ssm.sort_tif_files(ssm_dir)

    stacked_evi_data = stack_raster_months(sorted_evi_files)
    stacked_vpd_data = stack_raster_months(sorted_vpd_files)
    stacked_ssm_data = stack_raster_months(sorted_ssm_files)

    ssm._numpy_array_to_raster('ssm_plot.tif', stacked_ssm_data[:, :, 0],
                               [-126.75, 0.25, 0, 51.75, 0, -0.25],
                               'wgs84', gdal_data_type=gdal.GDT_Float32)
    ssm._numpy_array_to_raster('evi_plot.tif', stacked_evi_data[:, :, 0],
                               [-126.75, 0.25, 0, 51.75, 0, -0.25],
                               'wgs84', gdal_data_type=gdal.GDT_Float32)
    ssm._numpy_array_to_raster('vpd_plot.tif', stacked_vpd_data[:, :, 0],
                               [-126.75, 0.25, 0, 51.75, 0, -0.25],
                               'wgs84', gdal_data_type=gdal.GDT_Float32)

    main(
        ssm_data=stacked_ssm_data,
        evi_data=stacked_evi_data,
        vpd_data=stacked_vpd_data,
        data_start_date=start_date
        )
