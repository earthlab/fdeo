#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

@author: lukefanguna
"""
import numpy as np
from scipy.stats import norm


def compute_spi(md, sc):
    # Get the data for the time scale sc
    a1 = np.concatenate([md[i-1:len(md)-sc+i] for i in range(1, sc+1)], axis=1)
    y = np.sum(a1, axis=1)

    # Compute the SPI or SSI
    n = len(y)
    si = np.zeros((n, 1))

    month_range_end = n if n < 12 else 12

    for k in range(0, month_range_end):
        d = y[k::12]
        d = d.reshape((len(d), 1))
        si[k::12, 0] = empdis(d).flatten()

    si[:, 0] = norm.ppf(si[:, 0])

    return si.flatten()


def empdis(d):
    n = len(d)
    bp = np.zeros((n, 1))
    for i in range(n):
        bp[i, 0] = np.sum(d[:, 0] <= d[i, 0])

    y = (bp - 0.44) / (n + 0.12)

    return y


def data2index(resd, sc):
    lats, lons, n_months = resd.shape
    si = np.zeros((lats, lons, n_months))

    for i in range(lats):
        for j in range(lons):
            td = np.reshape(resd[i, j, :], (n_months, 1))
            si[i, j, :sc-1] = np.nan
            si[i, j, sc-1:] = compute_spi(td, sc)

    return si
