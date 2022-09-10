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


def calc_empirical_distribution(dist: Union[List, np.array]) -> np.array:
    """
    Computes the empirical distribution for list of input values
    Args:
        dist (List): Contains the input values for which to calculate the empirical distribution
    Returns:
        (np.array): Empirical distribution
    """

    n = len(dist)
    bp = np.zeros((n, 1))
    for i in range(n):
        bp[i] = sum(np.less_equal(dist, dist[i]))

    return np.divide((bp - 0.44), (n + 0.12))


def compute_spi(md, sc):
    """
    Gets the prep and smc data for the specific time scale then computes the Empirical drought index (SPI and SSI)
    from the data
    :param md:
    :param sc:
    :return:
    """
    # Get the data for the time scale sc
    evc = np.empty((132, 1))
    for i in range(sc):
        evc = md

    y = sum(evc)

    # Compute the SPI or SSI
    n = len(y)
    si = np.zeros((n, 1))

    for k in range(12):
        d = y[k - 1:n:12]
        si[k - 1:n:12] = calc_empirical_distribution(d)

    si[0][:] = ndtri(si[0][:])
    return si


def data2index(matrix: np.array, sc: int):
    """

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


def data2index_larger(matrx, sc):
    # matrx: matrix for the precipitation or soil moisture data

    # sc: scale of the index (>1, e.g., 3-month SPI or SSI)

    dimensions = np.shape(matrx)
    latitude = dimensions[0]  # x
    longitude = dimensions[1]  # y
    nmonths = dimensions[2]  # z

    # intialize the standard index matrix
    SI = np.empty((latitude, longitude, nmonths))

    for i in range(latitude):
        for j in range(longitude):
            np.shape(SI)
            td = np.zeros((nmonths, 1))
            td[:] = np.reshape(matrx[i][j][:], (nmonths, 1))

            SI[i][j][1:sc - 1] = math.nan
            SI[i][j][sc:(np.shape(SI))[2]] = compute_spi(td, sc)
    return SI
