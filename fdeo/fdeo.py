#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: lukefanguna
"""
import os
import argparse
import tempfile
import shutil
import numpy as np
from datetime import datetime
from scipy import signal
import matplotlib.pyplot as plt
from functions import data2index, empdis
from api import VPD, EVI, SSM
from utils import stack_raster_months, stack_arrays
from osgeo import gdal

FDEO_DIR = os.path.dirname(os.path.dirname(__file__))


def main(
       # ssm_prediction_data: np.array,
        #  evi_prediction_data: np.array,
        #    vpd_prediction_data: np.array
            ) -> None:
    """
    Reads input data and creates smoothed climatology of wildfire burned data. Default training period is 2003-2013.

    Args:
        ssm_data (np.array): Path to the ssm datafile. Defaults to 2003-2013 dataset.
        vpd_data (np.array): Path to the vpd datafile. Defaults to 2003-2013 dataset.
        evi_data (np.array): Path to the evi data. Defaults to 2003-2013 dataset.
    """

    # importing the land cover file (lc1.csv)
    lc1 = np.loadtxt(os.path.join(FDEO_DIR, 'data', 'lc1.csv'), delimiter=",")
    training_data_dimensions = lc1.shape
    training_data_months = 132
    """
    % ID for each Land cover type
    %1: Lake
    %2: Developed/Urban
    %3: Barren Land  
    %4: Deciduous
    %5: Evergreen
    %6: Mixed Forest
    %7: Shrubland
    %8: Herbaceous
    %9: Planted/Cultivated
    %10: Wetland
    %15: ocean
    """

    # soil moisture (sm) data from 2003-2013
    ssm_training_data = np.loadtxt(
        os.path.join(FDEO_DIR, 'data', 'sm_20032013.csv'), delimiter=","
    ).reshape((*training_data_dimensions, training_data_months))

    # enhanced vegetation index (EVI) data from 2003-2013
    evi_training_data = np.loadtxt(
        os.path.join(FDEO_DIR, 'data', 'EVI_20032013.csv'), delimiter=","
    ).reshape((*training_data_dimensions, training_data_months))

    # vapor pressure deficit (vpd) data from 2003-2013
    vpd_training_data = np.loadtxt(
        os.path.join(FDEO_DIR, 'data', 'vpd_20032013.csv'), delimiter=","
    ).reshape((*training_data_dimensions, training_data_months))

    # Fire data from 2003-2013
    firemon_tot_size = np.loadtxt(
        os.path.join(FDEO_DIR, 'data', 'firemon_tot_size.csv'), delimiter=",").reshape((*training_data_dimensions,
                                                                                        training_data_months))

    # calculate the fire climatology for each month

    # split the dim sizes for ease of use later on
    firemon_tot_size_x = training_data_dimensions[0]  # lat
    firemon_tot_size_y = training_data_dimensions[1]  # lon
    firemon_tot_size_z = training_data_months  # months

    firemon_tot_size_climatology = np.empty((firemon_tot_size_x, firemon_tot_size_y, 12))
    for i in range(firemon_tot_size_x):
        for j in range(firemon_tot_size_y):
            for k in range(12):
                firemon_tot_size_climatology[i][j][k] = np.mean(firemon_tot_size[i][j][k::12])

    # spatially smooth fire data using a 3 by 3 filter
    smooth_filter = np.ones((3, 3))

    firemon_tot_size_climatology_smoothed_3 = np.empty((firemon_tot_size_climatology.shape))
    for h in range(firemon_tot_size_climatology.shape[2]):
        firemon_tot_size_climatology_smoothed_3[:, :, h] = signal.convolve2d(
        firemon_tot_size_climatology[:, :, h], smooth_filter, mode='same'
    )

    firemon_tot_size_climatology_smoothed_3 /= smooth_filter.size

    # This part of the code creates a regression model for each LC type based on
    # the "best" drought indicator (DI) and then creates a historical record of
    # probabilistic and categorical wildfire prediction and observation data

    # TODO: Functions
    # Prepare and define training data sets
    # Deciduous DI
    sc_drought = 1  # month range
    deciduous_best_ba_training = data2index(ssm_training_data, sc_drought)

    # Shrubland DI
    sc_drought = 1
    shrubland_best_ba_training = data2index(evi_training_data, sc_drought)

    # Evergreen DI
    sc_drought = 3
    vpd_new_drought = data2index(vpd_training_data, sc_drought)
    evergreen_best_ba_training = vpd_new_drought

    # Herbaceous DI
    herbaceous_best_ba_training = vpd_new_drought

    # Wetland DI
    sc_drought = 3
    wetland_best_ba_training = data2index(ssm_training_data, sc_drought)

    # TODO: If prediction data is None then use the training files

    # Prepare and define prediction data sets
    # Deciduous DI
    # sc_drought = 1  # month range
    # deciduous_best_ba_prediction = data2index(ssm_prediction_data, sc_drought)

    # # Shrubland DI
    # sc_drought = 1
    # shrubland_best_ba_prediction = data2index(evi_prediction_data, sc_drought)

    # # Evergreen DI
    # sc_drought = 3
    # vpd_new_drought = data2index(vpd_prediction_data, sc_drought)
    # evergreen_best_ba_prediction = vpd_new_drought

    # # Herbaceous DI
    # herbaceous_best_ba_prediction = vpd_new_drought

    # # Wetland DI
    # sc_drought = 3
    # wetland_best_ba_prediction = data2index(ssm_prediction_data, sc_drought)

    # # Build the prediction model. One model for each land cover type

    # # Set initial dimensions of prediction probabilistic matrix
    # prediction_lat_size = ssm_prediction_data.shape[0]
    # prediction_lon_size = ssm_prediction_data.shape[1]
    # prediction_month_size = ssm_prediction_data.shape[2]

    # TODO: EV: This is what is actually used to hold the results. Need to change dimensions of this to fit the
    #  input data dimensions
    fire_pred_ini = np.full(ssm_training_data.shape, np.nan)

    # List of land-cover IDs according to below
    # 4: Deciduous
    # 5: Evergreen
    # 7: Shrubland
    # 8: Herbaceous
    # 10: Wetland
    lctypemat = [4, 5, 7, 8, 10]
    land_cover_training_data = [
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

    # land_cover_prediction_data = [
    #     {
    #         'data': deciduous_best_ba_prediction,
    #         'index': 4
    #     },
    #     {
    #         'data': evergreen_best_ba_prediction,
    #         'index': 5
    #     },
    #     {
    #         'data': shrubland_best_ba_prediction,
    #         'index': 7
    #     },
    #     {
    #         'data': herbaceous_best_ba_prediction,
    #         'index': 8
    #     },
    #     {
    #         'data': wetland_best_ba_prediction,
    #         'index': 10
    #     }
    # ]
    # Initialize the model and observation result arrays
    model_res = []
    obs_res = []
    for lc_forecast in land_cover_training_data:
        # First build the regression model for the LC Type initial parameters
        mat = np.empty((2, firemon_tot_size.size), dtype=float)  # Initial array
        m = 0  # Burned area for each LC. 1-5 is each diff one
        lead = 1  # Lead time
        for i in range(firemon_tot_size_x):
            for j in range(firemon_tot_size_y):
                if lc1[i][j] == lc_forecast['index']:
                    for k in range(firemon_tot_size_z):
                        # Leave the first 3-month empty to account for 2-month lead model development
                        if k - lead < 0:
                            mat[0, m] = np.nan
                        else:
                            mat[0, m] = lc_forecast['data'][i, j, k - lead]
                            mat[1, m] = firemon_tot_size[i, j, k]
                        m += 1

        # Remove NaN values from the data
        # Find indices of NaN values in each column
        idx_nan_1 = np.isnan(mat[0, :])
        idx_nan_2 = np.isnan(mat[1, :])

        # Combine indices of NaN values
        idx_nan = np.logical_or(idx_nan_1, idx_nan_2)

        # Filter the array to remove rows with NaN values
        mat = np.vstack([mat[0, :][~idx_nan], mat[1, :][~idx_nan]])
        print(mat.shape)

        # Bar plots to derive the regression model
        # Define number of bins
        bin = 10

        # Derive min and max of DI (drought indicator) data
        min_1 = np.min(mat[0, :])
        max_1 = np.max(mat[0, :])

        # Derive vector of bins based on DI data
        varbin = np.linspace(min_1, max_1, num=bin)

        # Initialize arrays to store results
        sample_size = np.zeros(bin)
        fire_freq_ave = np.zeros(bin)
        prob = np.zeros(bin)

        # Find observations in each bin
        k = 0
        for i in range(bin-1):
            # Find observations in each bin
            idx3 = np.where((mat[0, :] >= varbin[i]) & (mat[0, :] <= varbin[i+1]))

            # Find DI in each bin
            # Get corresponding burned area in each bin
            fire_freq_range = mat[0, :][idx3]
            # Calculate number of observations in each bin
            sample_size[k] = len(idx3[0])
            # Calculate sum burned area in each bin
            fire_freq_ave[k] = np.sum(fire_freq_range)
            # Calculate probability of fire at each bin
            prob[k] = fire_freq_ave[k] / sample_size[k]
            k += 1

        # Develop linear regression model
        x = np.array(varbin)
        y = np.array(prob)

        # Remove NaN from the observation input
        idx_nan = np.isnan(x)
        x = x[~idx_nan]
        y = y[~idx_nan]
        print(x, y)

        # Fit the regression model
        gofmat1 = np.polyfit(x, y, 2)

        # Calculate model and observation values at each bin. Each row represents one LC Type
        model_res.append(gofmat1[0] * (varbin ** 2) + gofmat1[1] * varbin + gofmat1[2])
        obs_res.append(prob)

        x = None
        y = None

        # Now build a historical forecast matrix based on the developed regression model for each LC Type
        for k in range(3, firemon_tot_size.shape[2]):
            for i in range(firemon_tot_size.shape[0]):
                for j in range(firemon_tot_size.shape[1]):
                    if lc1[i, j] == lc_forecast['index']:
                        fire_pred_ini[i, j, k] = (
                            gofmat1[0] * (lc_forecast['data'][i, j, k - lead] ** 2)
                            + gofmat1[1] * lc_forecast['data'][i, j, k - lead]
                            + gofmat1[2]
                        )

    # Build a correlation matrix and R2 matrix of goodness of fit for all models. Each row represents one LC Type
    corr_vector = np.zeros(5)
    r2_vector = np.zeros(5)

    for i in range(len(model_res)):
        # Remove NaN from observation
        idx_nan_4 = np.isnan(obs_res[i])
        model_res1 = model_res[i]
        obs_res1 = obs_res[i]
        model_res1 = model_res1[~idx_nan_4]
        obs_res1 = obs_res1[~idx_nan_4]

        # Calculate correlation
        corr_mat = np.corrcoef(model_res1, obs_res1)

        # Correlation vector of observation and model
        corr_vector[i] = corr_mat[0, 1]
        # R2 vector of observation and model
        r2_vector[i] = corr_vector[i] ** 2

    # We now calculate anomalies for observation and predictions

    # Observation data
    fire_obs_ini = firemon_tot_size

    # Subtract prediction and observation from climatology to derive anomalies
    for i in range(0, fire_obs_ini.shape[2], 12):
        fire_obs_ini[:, :, i:i+11] = fire_obs_ini[:, :, i:i+11] - firemon_tot_size_climatology_smoothed_3
        fire_pred_ini[:, :, i:i+11] = fire_pred_ini[:, :, i:i+11] - firemon_tot_size_climatology_smoothed_3

    # TODO: Need filtering of the same size
    # Prediction data might not be in 1 year chunks
    # fire_pred_ini_cate = np.empty((prediction_lat_size, prediction_lon_size, prediction_month_size))
    # pred_ini_split = np.dsplit(fire_pred_ini, prediction_month_size)
    # for i in range(prediction_month_size):
    #     j = i - (12 * (i // 12))
    #     np.append(fire_pred_ini_cate, pred_ini_split[i] - firemon_tot_size_climatology_smoothed_3[j], axis=2)

    # Derive bias adjusted observation and prediction probabilities and categorical forecast for the entire time series
    # distribution of prediction and observation come from Gringorten empirical distribution function (empdis function)

    # This section derives CDF of observation and prediction anomalies
    # derive CDF for each land cover type and for each month.
    # Build probabilistic prediction and observation matrices

    # This section derives CDF of observation and prediction anomalies
    # derive CDF for each land cover type and for each month. 
    # Build probabilistic prediction and observation matrices

    val_new_obs_tot_1 = np.zeros_like(fire_obs_ini)
    val_new_pred_tot_1 = np.zeros_like(fire_pred_ini)
    for k in range(fire_pred_ini.shape[2]):
        # matrix of observation and prediction anomalies for each month
        val_new_obs = fire_obs_ini[:, :, k]
        val_new_pred = fire_pred_ini[:, :, k]

        # derive CDF for each LC type
        for lc_co in range(len(lctypemat)):
            # derive observation and prediction anomalies for each LC Type
            idx_lc = np.where(lc1 == lctypemat[lc_co])
            mat = val_new_obs[idx_lc]
            mat1 = val_new_pred[idx_lc]

            # observation CDF
            y = empdis(mat)
            # prediction CDF
            y1 = empdis(mat1)

            val_new_obs[idx_lc] = y
            val_new_pred[idx_lc] = y1

        # build matrix of CDFs (probabilistic prediction and observation matrices)
        val_new_obs_tot_1[:, :, k] = val_new_obs
        val_new_pred_tot_1[:, :, k] = val_new_pred


    np.savetxt(f'fire_obs_ini_cate{fire_obs_ini.shape}.txt', fire_obs_ini.reshape(
        -1, fire_obs_ini.shape[-1]))
    np.savetxt(f'val_new_obs_tot_1_{val_new_obs_tot_1.shape}.txt', val_new_obs_tot_1.reshape(
        -1, val_new_obs_tot_1.shape[-1]))
    np.savetxt(f'val_new_pred_tot_1_{val_new_pred_tot_1.shape}.txt', val_new_pred_tot_1.reshape(
        -1, val_new_pred_tot_1.shape[-1]))

    val_new_obs_tot_2 = np.zeros_like(fire_obs_ini)
    val_new_pred_tot_2 = np.zeros_like(fire_pred_ini)
    # Build categorical prediction and observation matrices
    # derive categorical prediction and observation data for each month

    for k in range(fire_pred_ini.shape[2]):
        # matrix of observation and prediction anomalies for each month
        val_new_obs = fire_obs_ini[:, :, k]
        val_new_pred = fire_pred_ini[:, :, k]

        # build a loop for each LC type
        for lc_co in range(len(lctypemat)):
            # derive observation and prediction anomalies for each LC Type
            idx_lc = np.where(lc1 == lctypemat[lc_co])
            mat = val_new_obs[idx_lc]
            mat1 = val_new_pred[idx_lc]

            # observation CDF
            # 33 percentile threshold for observation time series
            y1 = empdis(mat)
            T1 = np.min(y1)
            T2 = np.max(y1)
            T3 = (T2 - T1) / 3
            T4 = T1 + T3
            T5 = y1 - T4
            T6 = np.abs(T5)
            T7 = np.where(T6 == np.min(T6))
            below_no_obs = mat[T7[0][0]]

            # 66 percentile threshold for observation time series
            T9 = T4 + T3
            T10 = y1 - T9
            T11 = np.abs(T10)
            T12 = np.where(T11 == np.min(T11))
            above_no_obs = mat[T12[0][0]]

            # prediction CDF
            # 33 percentile threshold for prediction time series
            y1 = empdis(mat1)
            T1 = np.min(y1)
            T2 = np.max(y1)
            T3 = (T2 - T1) / 3
            T4 = T1 + T3
            T5 = y1 - T4
            T6 = np.abs(T5)
            T7 = np.where(T6 == np.min(T6))
            below_no_pred = mat1[T7[0][0]]

            # 66 percentile threshold for prediction time series
            T9 = T4 + T3
            T10 = y1 - T9
            T11 = np.abs(T10)
            T12 = np.where(T11 == np.min(T11))
            above_no_pred = mat1[T12[0][0]]

            # populate categorical observation matrix
            for i in range(fire_obs_ini.shape[0]):
                for j in range(fire_obs_ini.shape[1]):
                    if lc1[i, j] == lctypemat[lc_co] and val_new_obs[i, j] < below_no_obs:
                        val_new_obs[i, j] = -1
                    elif lc1[i, j] == lctypemat[lc_co] and val_new_obs[i, j] > above_no_obs:
                        val_new_obs[i, j] = 1
                    elif lc1[i, j] == lctypemat[lc_co] and below_no_obs <= val_new_obs[i, j] <= above_no_obs:
                        val_new_obs[i, j] = 0

            # populate categorical prediction matrix
            for i in range(fire_pred_ini.shape[0]):
                for j in range(fire_pred_ini.shape[1]):
                    if lc1[i, j] == lctypemat[lc_co] and val_new_pred[i, j] < below_no_pred:
                        val_new_pred[i, j] = -1
                    elif lc1[i, j] == lctypemat[lc_co] and val_new_pred[i, j] > above_no_pred:
                        val_new_pred[i, j] = 1
                    elif lc1[i, j] == lctypemat[lc_co] and below_no_pred <= val_new_pred[i, j] <= above_no_pred:
                        val_new_pred[i, j] = 0

        # categorical prediction and observation final matrices
        val_new_obs_tot_2[:, :, k] = val_new_obs
        val_new_pred_tot_2[:, :, k] = val_new_pred

    np.savetxt(f'val_new_obs_tot_2_{val_new_obs_tot_2.shape}.txt', val_new_obs_tot_2.reshape(
        -1, val_new_obs_tot_2.shape[-1]))
    np.savetxt(f'val_new_pred_tot_2_{val_new_pred_tot_2.shape}.txt', val_new_pred_tot_2.reshape(
        -1, val_new_pred_tot_2.shape[-1]))


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
    if args.start_date is not None or args.end_date is not None:
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

        ssm = SSM(username=username, password=password)
        evi = EVI(username=username, password=password)
        vpd = VPD(username=username, password=password)

        tempdir = tempfile.mkdtemp(prefix='fdeo')
        ssm_dir = os.path.join(tempdir, 'ssm')
        evi_dir = os.path.join(tempdir, 'evi')
        vpd_dir = os.path.join(tempdir, 'vpd')
        os.makedirs(ssm_dir)
        os.makedirs(evi_dir)
        os.makedirs(vpd_dir)
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date is not None else None
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date is not None else None
        ssm_data = ssm.create_clipped_time_series(ssm_dir, start_date, end_date)
        evi_data = evi.create_clipped_time_series(evi_dir, start_date, end_date)
        vpd_data = vpd.create_clipped_time_series(vpd_dir, start_date, end_date)

        # Sample the EVI and VPD data to the SSM 0.25 deg spatial resolution
        ssm_sample_file = os.path.join(ssm_dir, os.listdir(ssm_dir)[0])

        sorted_evi_files = evi.sort_tif_files(evi_dir)
        sorted_vpd_files = vpd.sort_tif_files(vpd_dir)

        print(sorted_evi_files)
        print(sorted_vpd_files)
        print(ssm.sort_tif_files(ssm_dir))

        stacked_evi_data = stack_raster_months(sorted_evi_files)
        stacked_vpd_data = stack_raster_months(sorted_vpd_files)

        print('stacked')

        ssm_month_data = []
        for ssm_file in ssm.sort_tif_files(ssm_dir):
            ssm_file_obj = gdal.Open(ssm_file)
            ssm_month_data.append(ssm_file_obj.GetRasterBand(1).ReadAsArray())

        stacked_ssm_data = stack_arrays(ssm_month_data)

        print(stacked_ssm_data.shape)
        print(stacked_evi_data.shape)
        print(stacked_vpd_data.shape)

        # TODO: Just for testing
        ssm._numpy_array_to_raster('ssm_plot.tif', np.flipud(stacked_ssm_data[:,:,0]), [-126.75, 0.25, 0, 51.75, 0, -0.25],
                                   'wgs84', gdal_data_type=gdal.GDT_Float32)
        ssm._numpy_array_to_raster('evi_plot.tif', np.flipud(stacked_evi_data[:,:,0]), [-126.75, 0.25, 0, 51.75, 0, -0.25],
                                   'wgs84', gdal_data_type=gdal.GDT_Float32)
        ssm._numpy_array_to_raster('vpd_plot.tif', np.flipud(stacked_vpd_data[:,:,0]), [-126.75, 0.25, 0, 51.75, 0, -0.25],
                                   'wgs84', gdal_data_type=gdal.GDT_Float32)
    print('Opening')
    # stacked_ssm_data = gdal.Open(args.ssm_test)
    # stacked_ssm_data = stacked_ssm_data.GetRasterBand(1).ReadAsArray().reshape((112, 244, 1))

    # stacked_evi_data = gdal.Open(args.evi_test)
    # stacked_evi_data = stacked_evi_data.GetRasterBand(1).ReadAsArray().reshape((112, 244, 1))
    # stacked_vpd_data = gdal.Open(args.vpd_test)
    # stacked_vpd_data = stacked_vpd_data.GetRasterBand(1).ReadAsArray().reshape((112, 244, 1))
    print('Opened')

    main(
        # ssm_prediction_data=stacked_ssm_data,
        #   evi_prediction_data=stacked_evi_data,
        #  vpd_prediction_data=stacked_vpd_data
         )

    # if stacked_ssm_data is not None:
    #     print(ssm_dir)
    #     # shutil.rmtree(ssm_dir)
    # if stacked_evi_data is not None:
    #     print(evi_dir)
    #     # shutil.rmtree(evi_dir)
    # if stacked_vpd_data is not None:
    #     print(vpd_dir)
    #     # shutil.rmtree(vpd_dir)
