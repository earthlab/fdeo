#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: lukefanguna
"""
from typing import List, Union
import numpy as np
import math
from scipy.special import ndtri


def calc_plotting_position(dist: Union[List, np.array]) -> np.array:
    """
    Computes the Gringorten plotting position
    Args:
        dist (List): Contains the data for which to calculate plotting positions
    Returns:
        (np.array): Plotting positions
    """

    n = len(dist)
    bp = np.zeros((n, 1))
    for i in range(n):
        bp[i] = sum(np.less_equal(dist, dist[i]))

    return np.divide((bp - 0.44), (n + 0.12))


def compute_spi(md, sc):
    """
    Gets the prep and smc data for the specific timescale then computes the Empirical drought index (SPI and SSI)
    from the data
    :param md:
    :param sc:
    :return:
    """
    # Get the data for the timescale sc
    evc = np.empty((132, 1))
    for i in range(sc):
        evc = md

    y = sum(evc)

    # Compute the SPI or SSI
    n = len(y)
    si = np.zeros((n, 1))

    for k in range(12):
        d = y[k - 1:n:12]
        si[k - 1:n:12] = calc_plotting_position(d)

    si[0][:] = ndtri(si[0][:])
    return si


def data2index(matrix: np.array, sc: int):
    """
    Derives Drought Index of SSM and EVI data
    Args:
        matrix (np.array): Input array containing the precipitation or soil moisture data
        sc (int): scale of the index (>1, e.g., 3-month SPI or SSI)
    Returns:
        si (np.array):
    """

    dimensions = np.shape(matrix)
    latitude = dimensions[0]  # x
    longitude = dimensions[1]  # y
    n_months = dimensions[2]  # z

    # initialize the standard index matrix
    si = np.ones((latitude, longitude, n_months))

    for i in range(latitude):
        for j in range(longitude):
            td = np.zeros((n_months, 1))
            td[:] = np.reshape(matrix[i][j][:], (n_months, 1))

            si[i][j][sc - 1:] = float("NaN")
            si[i][j][sc:(np.shape(si))[2]] = compute_spi(td, sc)
    return si


def data2index_larger(matrix: np.array, sc: int):
    """
    Derives Drought Index of VPD data
    Args:
        matrix (np.array): Input array containing the vapor pressure deficit data
        sc (int): scale of the index (>1, e.g., 3-month SPI or SSI)
    """
    dimensions = np.shape(matrix)
    latitude = dimensions[0]  # x
    longitude = dimensions[1]  # y
    n_months = dimensions[2]  # z

    # initialize the standard index matrix
    si = np.empty((latitude, longitude, n_months))

    for i in range(latitude):
        for j in range(longitude):
            np.shape(si)
            td = np.zeros((n_months, 1))
            td[:] = np.reshape(matrix[i][j][:], (n_months, 1))

            si[i][j][1:sc - 1] = math.nan
            si[i][j][sc:(np.shape(si))[2]] = compute_spi(td, sc)
    return si
