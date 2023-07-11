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
    a1 = np.column_stack([md[i:len(md)-sc+i+1] for i in range(sc)])
    Y = np.sum(a1, axis=1)
    
    # Compute the SPI or SSI
    n = len(Y)
    si = np.zeros((n, 1))
    
    for k in range(12):
        d = Y[k:12:n]
        si[k:12:n, 0] = empdis(d)
    
    si[:, 0] = norm.ppf(si[:, 0])
    
    return si.flatten()


def empdis(d):
    n = len(d)
    bp = np.zeros(n)

    for i in range(n):
        bp[i] = np.sum(d <= d[i])

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
