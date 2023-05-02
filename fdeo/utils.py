from typing import List
from osgeo import gdal
from scipy import interpolate
import numpy as np
import os


def fix_resolution_and_stack(sorted_input_files: List[str], target_resolution_file: str):
    month_data = []
    for file in sorted_input_files:
        data = set_tiff_resolution(file, target_resolution_file)
        month_data.append(data)
    return np.stack(month_data, axis=2)


def set_tiff_resolution(input_resolution_path: str, target_resolution_path: str):
    input_res = gdal.Open(input_resolution_path)

    # Access the data
    input_res_band = input_res.GetRasterBand(1)
    input_res_data = input_res_band.ReadAsArray()

    input_res_lons, input_res_lats = get_geo_locations_from_tif(input_res)

    input_res_interp = interpolate.interp2d(input_res_lons, input_res_lats, input_res_data, kind='linear')

    target_res = gdal.Open(target_resolution_path)

    target_res_lons, target_res_lats = get_geo_locations_from_tif(target_res)
    target_res_data = input_res_interp(target_res_lons, target_res_lats)

    return target_res_data


def get_geo_locations_from_tif(raster):
    # Get geolocation information
    geo_transform = raster.GetGeoTransform()
    x_size = geo_transform[1]
    y_size = geo_transform[5]
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]

    # Get geolocation of each data point
    lats = []
    for row in range(raster.RasterYSize):
        lats.append(y_origin - (row * y_size))
    res_lats = np.repeat(np.array([lats]), raster.RasterXSize, axis=0)

    lons = []
    for col in range(raster.RasterXSize):
        lons.append(x_origin + (col * x_size))
    res_lons = np.repeat(np.array([lons]), raster.RasterYSize, axis=0)

    return res_lons, res_lats
