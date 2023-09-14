#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 13:13:16 2022

"""
import numpy as np
from scipy.stats import norm


def compute_spi(md, sc):
    a1 = np.concatenate([md[i:len(md)-sc+i] for i in range(sc+1)], axis=1)
    y = np.sum(a1, axis=1)

    # Compute the SPI or SSI
    n = len(y)
    si = np.zeros(n)

    month_range_end = n if n < 12 else 12

    for k in range(0, month_range_end):
        d = y[k::12]
        si[k::12] = empdis(d)

    si = norm.ppf(si)

    return si


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
            si[i, j, :sc] = np.nan
            si[i, j, sc:] = compute_spi(td, sc)

    return si
