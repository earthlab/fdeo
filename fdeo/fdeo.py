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
from functions import data2index, data2index_larger, calc_plotting_position
from api import VPD, EVI, SSM
from utils import fix_resolution_and_stack
from osgeo import gdal

FDEO_DIR = os.path.dirname(os.path.dirname(__file__))


def main(ssm_data: np.array = None, evi_data: np.array = None, vpd_data: np.array = None) -> None:
    """
    Reads input data and creates smoothed climatology of wildfire burned data. Default training period is 2003-2013.

    Args:
        ssm_data (np.array): Path to the ssm datafile. Defaults to 2003-2013 dataset.
        vpd_data (np.array): Path to the vpd datafile. Defaults to 2003-2013 dataset.
        evi_data (np.array): Path to the evi data. Defaults to 2003-2013 dataset.
    """

    # importing the land cover file (lc1.csv)
    lc1 = np.loadtxt(os.path.join(FDEO_DIR, 'data', 'lc1.csv'), delimiter=",")
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

    if ssm_data is None:
        # soil moisture (sm) data from 2003-2013
        ssm_data = np.loadtxt(
            os.path.join(FDEO_DIR, 'data', 'sm_20032013.csv') if ssm_data is None else ssm_data, delimiter=","
        ).reshape((112, 244, 132))

    if evi_data is None:
        # enhanced vegetation index (EVI) data from 2003-2013
        evi_data = np.loadtxt(
            os.path.join(FDEO_DIR, 'data', 'EVI_20032013.csv') if evi_data is None else evi_data, delimiter=","
        ).reshape((112, 244, 132))

    if vpd_data is None:
        # vapor pressure deficit (vpd) data from 2003-2013
        vpd_data = np.loadtxt(
            os.path.join(FDEO_DIR, 'data', 'vpd_20032013.csv') if vpd_data is None else vpd_data, delimiter=","
        ).reshape((112, 244, 132))

    # Fire data from 2003-2013
    firemon_tot_size = np.loadtxt(
        os.path.join(FDEO_DIR, 'data', 'firemon_tot_size.csv'), delimiter=",").reshape((112, 244, 132))

    # calculate the fire climatology for each month

    # split the dim sizes for ease
    mtrxshape = np.shape(firemon_tot_size)
    firemon_tot_size_x = mtrxshape[0]
    firemon_tot_size_y = mtrxshape[1]
    firemon_tot_size_z = mtrxshape[2]

    firemon_tot_size_climatology = np.empty((firemon_tot_size_x, firemon_tot_size_y, 12))
    for i in range(firemon_tot_size_x):
        for j in range(firemon_tot_size_y):
            for k in range(12):
                firemon_tot_size_climatology[i][j][k] = np.mean(firemon_tot_size[i][j][k:firemon_tot_size_z:12])

    # spatially smooth fire data using a 3 by 3 filter
    smooth_filter = np.ones((3, 3))

    firemon_tot_size_climatology_smoothed_3 = np.empty((firemon_tot_size_x, firemon_tot_size_y, 12))
    for h in range(len(firemon_tot_size_climatology)):
        rotating_arr = firemon_tot_size_climatology[:][:][h]
        rotator = np.rot90(rotating_arr)
        firemon_tot_size_climatology_smoothed_3[:][:][h] = signal.convolve2d(firemon_tot_size_climatology[:][:][h],
                                                                             rotator,
                                                                             mode='same')

    firemon_tot_size_climatology_smoothed_3 = firemon_tot_size_climatology_smoothed_3 / (
            len(smooth_filter[0]) * len(smooth_filter[1]))

    # This part of the code creates a regression model for each LC type based on
    # the "best" drought indicator (DI) and then creates a historical record of
    # probabilistic and categorical wildfire prediction and observation data

    # Deciduous DI
    sc_drought = 1  # month range
    deciduous_best_ba = data2index(ssm_data, sc_drought)

    # Shrubland DI
    sc_drought = 1
    shrubland_best_ba = data2index(evi_data, sc_drought)

    # Evergreen DI
    sc_drought = 3
    vpd_new_drought = data2index_larger(vpd_data, sc_drought)
    evergreen_best_ba = vpd_new_drought

    # Herbaceous DI
    herbaceous_best_ba = vpd_new_drought

    # Wetland DI
    sc_drought = 3
    wetland_best_ba = data2index(ssm_data, sc_drought)

    # Build the prediction model. One model for each land cover type

    # Set initial dimensions of prediction probabilistic matrix
    fire_pred_ini = np.empty((firemon_tot_size_x, firemon_tot_size_y, firemon_tot_size_z))

    # List of land-cover IDs according to below
    # 4: Deciduous
    # 5: Evergreen
    # 7: Shrubland
    # 8: Herbaceous
    # 10: Wetland
    lctypemat = [4, 5, 7, 8, 10]
    land_cover_data = [
        {
            'data': deciduous_best_ba,
            'index': 4
        },
        {
            'data': evergreen_best_ba,
            'index': 5
        },
        {
            'data': shrubland_best_ba,
            'index': 7
        },
        {
            'data': herbaceous_best_ba,
            'index': 8
        },
        {
            'data': wetland_best_ba,
            'index': 10
        }
    ]
    for lc_co, lc_forecast in enumerate(land_cover_data):
        # First build the regression model for the LC Type initial parameters
        mat = np.empty((2, firemon_tot_size.size), dtype=float)  # Initial array
        m = 0  # Burned area for each LC. 1-5 is each diff one
        lead = 1  # Lead time
        for i in range(firemon_tot_size_x):
            for j in range(firemon_tot_size_y):
                if lc1[i][j] == lc_forecast['index']:
                    for k in range(firemon_tot_size_z):
                        # Leave the first 3-month empty to account for 2-month lead model development
                        if k - lead < 1:
                            mat[0][m] = np.nan
                        else:
                            # Drought index
                            mat[0][m] = lc_forecast['data'][i][j][k - lead]

                            # Fire burned area
                            mat[1][m] = firemon_tot_size[i][j][k]
                        m += 1

        di_mat = mat[0]
        ba_mat = mat[1]
        np.nan_to_num(di_mat, False, 999)
        np.nan_to_num(ba_mat, False, 999)
        idx_nan_1 = np.where(mat[0] == 999)
        idx_nan_2 = np.where(mat[1] == 999)
        idx_nan_3 = np.concatenate((idx_nan_1, idx_nan_2), axis=None)  # Combines NaN values/stacks them
        idx_nan_4 = np.unique(idx_nan_3)
        mat = np.column_stack((di_mat, ba_mat))
        mat = np.delete(mat, idx_nan_4, axis=0)

        # Derive vector of bins based on DI data
        varbin = np.arange(-1.6414, 1.9697, .3283)  # -1.6414->1.6414 in inc of .3283

        # Find observations in each bin
        k = 0
        sample_size = np.empty((11, 1))
        fire_freq_ave = np.empty((11, 1))
        prob = np.empty((11, 1))

        for i in range(len(varbin)):
            # Find observations in each bin
            idx3 = np.less_equal(np.logical_and(mat[0] >= varbin[i], mat[0]),
                                 (varbin[i] + ((np.max(mat[0]) - np.min(mat[0])) / 2)))

            # Get corresponding burned area in each bin
            fire_freq_range = mat[1][idx3]

            # Calculate number of observations in each bin
            sample_size[k] = (len(idx3))

            # Calculate sum burned area in each bin
            fire_freq_ave[k] = sum(fire_freq_range)

            # Calculate probability of fire at each bin
            prob[k] = (fire_freq_ave[k] / sample_size[k])
            k += 1

        # Develop linear regression model
        x = varbin
        y = prob

        idx_nan = np.isnan(y)

        # Take out nan values in it
        x = x[~idx_nan.flatten()]
        y = y[~idx_nan.flatten()]

        # Fit the regression model
        gofmat1 = np.polyfit(x, y, 2)

        # Calculate model and observation values at each bin. Each row represents one LC Type
        model_res = np.empty((lc_co + 1, 11))
        obs_res = np.empty((lc_co + 1, 11))
        for i in range(len(varbin)):
            # Model at each bin
            model_res[lc_co][i] = gofmat1[1] * (varbin[i] ** 2) + gofmat1[1] * varbin[i] + gofmat1[2]

            # Observation at each bin
            obs_res[lc_co][i] = prob[i]

        del x
        del y

        # Now build a historical forecast matrix based on the developed regression model for each LC Type
        for k in range(4, firemon_tot_size_z):
            for i in range(firemon_tot_size_x):
                for j in range(firemon_tot_size_y):
                    if lc1[i][j] == lc_forecast['index']:
                        fire_pred_ini[i][j][k] = gofmat1[0] * (lc_forecast['data'][i][j][k - lead] ** 2) + gofmat1[1] \
                                                 * lc_forecast['data'][i][j][k - lead] + gofmat1[2]

        # Build a correlation matrix and R2 matrix of goodness of fit for all models. Each row represents one LC Type
        forrange = np.shape(model_res)
        for i in range(forrange[0]):
            pmvec = model_res[i]  # Creates separate matrix
            povec = obs_res[i]  # Creates separate matrix

            # Remove nan from observation
            idx_nan_4 = ~(obs_res[i][:] == 999)  # Finds NaN values
            pmvec = pmvec[idx_nan_4]  # Removes nan values
            povec = povec[idx_nan_4]  # Removes nan values

            # Calculate correlation
            corr_mat = np.corrcoef(pmvec, povec)
            corr_mat = np.nan_to_num(corr_mat, False, 999)

            # Correlation vector of observation and model
            corr_vector = np.shape(model_res)
            corr_vector = np.empty(corr_vector[0])
            corr_vector[i] = corr_mat[0][1]

            # R2 vector of observation and model
            r2_vector = np.shape(model_res)
            r2_vector = np.arange(r2_vector[0])
            r2_vector[i] = corr_vector[i] ** 2

    # We now calculate anomalies for observation and predictions

    # Observation data
    fire_obs_ini = firemon_tot_size

    # subtract prediction and observation from climatology to derive anomalies
    fire_pred_ini_cate = np.empty((firemon_tot_size_x, firemon_tot_size_y, firemon_tot_size_z))
    fire_obs_ini_cate = np.empty((firemon_tot_size_x, firemon_tot_size_y, firemon_tot_size_z))

    obs_ini_split = np.dsplit(fire_obs_ini, 11)
    pred_ini_split = np.dsplit(fire_pred_ini, 11)
    for i in range(11):
        np.append(fire_obs_ini_cate, obs_ini_split[i] - firemon_tot_size_climatology_smoothed_3, axis=2)
        np.append(fire_pred_ini_cate, pred_ini_split[i] - firemon_tot_size_climatology_smoothed_3, axis=2)

    # Derive bias adjusted observation and prediction probabilities and categorical forecast for the entire time series
    # distribution of prediction and observation come from Gringorten empirical distribution function (empdis function)

    # This section derives CDF of observation and prediction anomalies
    # derive CDF for each land cover type and for each month.
    # Build probabilistic prediction and observation matrices
    val_new_obs_tot_1 = np.empty((firemon_tot_size_x, firemon_tot_size_y, firemon_tot_size_z))
    val_new_pred_tot_1 = np.empty((firemon_tot_size_x, firemon_tot_size_y, firemon_tot_size_z))
    obs_split = np.dsplit(fire_obs_ini_cate, 132)
    pred_split = np.dsplit(fire_pred_ini_cate, 132)
    dimensions = np.shape(fire_pred_ini_cate)

    for k in range(dimensions[0]):

        # Matrix of observation and prediction anomalies for each month
        val_new_obs = obs_split[k]
        val_new_pred = pred_split[k]

        # Derive CDF for each LC type
        for lc_type in lctypemat:

            # Derive observation and prediction anomalies for each LC Type
            idx_lc = np.equal(lc1, lc_type)  # Creates a 1d array that fulfills cond

            mat = val_new_obs[idx_lc]  # Picks values from val_new_obs that fulfills cond

            # Observation CDF
            y = calc_plotting_position(mat)

            val_new_obs[idx_lc] = y
            val_new_pred[idx_lc] = y

        # Build matrix of CDFs (probabilistic prediction and observation matrices)

        val_new_obs_tot_1[k][:][:] = fire_obs_ini_cate[k][:][:]
        val_new_pred_tot_1[k][:][:] = fire_pred_ini_cate[k][:][:]

        # build a loop for each LC Type
        for lc_type in lctypemat:

            # derive observation and prediction anomalies for each LC Type
            idx_lc = np.equal(lc1, lc_type)  # creates a 1d array that fulfills cond

            # TODO: EV: Should this loop be nested in the above loop which defines val_new_obs?
            mat = val_new_obs[idx_lc]  # picks values from val_new_obs that fulfills cond
            mat1 = val_new_pred[idx_lc]  # picks values from val_new_pred that fulfills cond

            # TODO: Make this into a function
            # Observation CDF 33 percentile threshold for observation time series
            y1 = calc_plotting_position(mat)
            t1 = np.min(y1)
            t2 = np.max(y1)
            t3 = (t2 - t1) / 3
            t4 = t1 + t3
            t5 = y1 - t4
            t6 = abs(t5)
            t7 = np.where(t6 == np.min(t6))
            below_no_obs = mat[t7]
            below_no_obs = below_no_obs.flatten()
            # 66 percentile threshold for observation time series
            t9 = t4 + t3
            t10 = y1 - t9
            t11 = abs(t10)
            t12 = np.where(t11 == np.min(t11))
            above_no_obs = mat[t12]
            above_no_obs = above_no_obs.flatten()

            # prediction CDF
            # 33 percentile threshold for prediction time series
            y1 = calc_plotting_position(mat1)
            t1 = np.min(y1)
            t2 = np.max(y1)
            t3 = (t2 - t1) / 3
            t4 = t1 + t3
            t5 = y1 - t4
            t6 = abs(t5)
            t7 = np.where(t6 == min(t6))
            below_no_pred = mat1[t7]
            below_no_pred = below_no_pred.flatten()
            # 66 percentile threshold for prediction time series
            t9 = t4 + t3
            t10 = y1 - t9
            t11 = abs(t10)
            t12 = np.where(t11 == np.min(t11))
            above_no_pred = mat1[t12]
            above_no_pred = above_no_pred.flatten()

            # populate categorical observation matrix
            for i in range((np.shape(fire_obs_ini_cate))[0]):
                for j in range((np.shape(fire_obs_ini_cate))[1]):
                    if (lc1[i][j] == lc_type) & (val_new_obs[i][j][0] < below_no_obs).all():
                        val_new_obs[i][j] = -1
                    elif (lc1[i][j] == lc_type) & (val_new_obs[i][j][0] > above_no_obs).all():
                        val_new_obs[i][j] = 1
                    elif (lc1[i][j] == lc_type) & (val_new_obs[i][j][0] >= below_no_obs).all() & (
                            val_new_obs[i][j] <= above_no_obs).all():
                        val_new_obs[i][j] = 0

            # populate categorical prediction matrix
            for i in range((np.shape(fire_pred_ini_cate))[0]):
                for j in range((np.shape(fire_pred_ini_cate))[1]):
                    if (lc1[i][j] == lc_type) & (val_new_pred[i][j][0] < below_no_pred).all():
                        val_new_pred[i][j] = -1
                    elif (lc1[i][j] == lc_type) & (val_new_pred[i][j][0] > above_no_pred).all():
                        val_new_pred[i][j] = 1
                    elif (lc1[i][j] == lc_type) & (val_new_pred[i][j][0] >= below_no_pred).all() & (
                            val_new_pred[i][j] <= above_no_pred).all():
                        val_new_pred[i][j] = 0

    # FIG 6 abd 7 of the paper for aug 2013

    a = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # August for title of the plot
    figco = 8

    month_o_year = np.arange(0, (np.shape(val_new_pred_tot_1))[2], 11)
    year = 11

    # month to graph histograms
    mo = month_o_year[year] + 7

    # plot probabilities of observations
    val_split = np.dsplit(val_new_obs_tot_1, 132)
    val = val_split[mo - 1]
    val = val.reshape((112, 244))
    # exclude LC types out of the scope of the study
    for i in range(112):
        for j in range(244):
            if (lc1[i][k] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
                val[i][j] = float("NaN")

    val = np.rot90(val.T)
    fig, (fig1) = plt.subplots(1, 1)
    fig1.pcolor(val)
    plt.xlabel(a[figco])
    plt.show()


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
    ssm_data = None
    vpd_data = None
    evi_data = None

    args = parser.parse_args()
    if args.start_date is not None or args.end_date is not None:
        username = args.username
        password = args.password
        if args.credentials is not None:
            with open(args.credentials, 'r') as f:
                lines = f.readlines()
                username = lines[0].strip('\n').strip(' ')
                password = lines[1].strip('\n').strip(' ')

        if args.username is None or args.password is None:
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
        ssm_sample_file = os.listdir(ssm_dir)[0]

        sorted_evi_files = evi.sort_files(evi_dir)
        sorted_vpd_files = vpd.sort_files(vpd_dir)

        stacked_evi_data = fix_resolution_and_stack(sorted_evi_files, ssm_sample_file)
        stacked_vpd_data = fix_resolution_and_stack(sorted_vpd_files, ssm_sample_file)

        ssm_month_data = []
        for ssm_file in ssm.sort_files(ssm_dir):
            ssm_file_obj = gdal.Open(ssm_file)
            ssm_month_data.append(ssm_file_obj.GetRasterBand(1).ReadAsArray())
        stacked_ssm_data = np.stack(ssm_month_data, axis=2)

        print(stacked_ssm_data.shape)
        print(stacked_evi_data.shape)
        print(stacked_vpd_data.shape)

    main(ssm_data=stacked_ssm_data, evi_data=stacked_evi_data, vpd_data=stacked_vpd_data)

    if ssm_data is not None:
        shutil.rmtree(ssm_dir)
    if evi_data is not None:
        shutil.rmtree(evi_dir)
    if vpd_data is not None:
        shutil.rmtree(vpd_dir)
