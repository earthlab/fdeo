from pyhdf.SD import SD, SDC
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
from fdeo.api import BaseAPI


def validate(in_file: str):
    f = SD(in_file)
    burn_day = f.select('Burn Date').get()
    no_burn = np.where(burn_day <= 0)
    burn = np.where(burn_day > 0)
    burn_day[no_burn] = 0
    burn_day[burn] = 1
    plt.imshow(burn_day)
    plt.show()


def convert_to_tif(in_file: str, geo_transform):
    f = SD(in_file)
    burn_day = f.select('Burn Date').get()
    no_burn = np.where(burn_day <= 0)
    burn = np.where(burn_day > 0)
    burn_day[no_burn] = 0
    burn_day[burn] = 1
    BaseAPI._numpy_array_to_raster(in_file.replace('.hdf', '.tif'), burn_day, geo_transform, projection='wgs84')
