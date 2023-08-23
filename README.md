# 

# Overview
The Fire Danger from Earth Observations (FDEO) Python package incorporates three types of remote sensing data in order
to predict fire danger over all of CONUS. These data are Enhanced Vegatation Index (EVI), Vapor Pressure Deficit (VPD),
and Surface Soil Moisture (SSM). The model used to make predictions was trained using these data and fire burned area
data from the National Fire Program Analysis Fire-Occurrence Database (FPA FOD) from 2003 to 2013. A more thorough 
explanation of the methods and datasets can be found in the FDEO paper here: https://www.mdpi.com/2072-4292/12/8/1252

A two-month lead time is required for fire danger predictions. Therefore, for example, data from Aug 2020 is used to
make predictions for October 2020. Two types of outputs are produced: a categorical prediction 
(below normal, normal, and above normal danger) and a probability prediction between 0 and 1. Each of these scores is
produced at a monthly temporal and 0.25 x 0.25 deg spatial resolution. A .tif file for both categorical and probable 
scores is produced which contain a band for each month of prediction results.   

## Datasets

### Enhanced Vegetation Index (EVI)
The EVI data used for inference is from NASA's MODIS satellite hosted at https://e4ftl01.cr.usgs.gov/MOLT/MOD13C2.061/
The band used from these files was the monthly 0.05 deg EVI band 'CMG 0.05 Deg Monthly EVI'. All data was scaled down by
a factor of 1000 and all off-nominal values (-3000) converted to NaN before calculating drought indicators. The data was
sampled to a 0.25 x 0.25 deg spatial resolution using linear interpolation.

### Vapor Pressure Deficit (VPD)
The VPD data used for inference is derived from NASA's AIRS satellite hosted at https://acdisc.gesdisc.eosdis.nasa.gov/opendap/Aqua_AIRS_Level3/AIRS3STM.7.0/
VPD is calculated using the relative humidity ('RelHumSurf_A' band) and surface air temperature ('SurfAirTemp_A' band)
as follows:
```python
import numpy as np
from pyhdf.SD import SD, SDC

def calculate_vpd(vpd_file: str):
    """
    Args:
        vpd_file (str): Path to file AIRS data containing the surface temperature and humidity bands.
    """
    vpd = SD(vpd_file, SDC.READ)
    rel_hum = np.clip(vpd.select('RelHumSurf_A').get(), a_min=0, a_max=None)
    surf_temp = vpd.select('SurfAirTemp_A').get() - 273.15

    es = 0.611 * np.exp((17.27 * surf_temp) / (surf_temp + 237.3))  # saturation vapor pressure
    e = (rel_hum / 100) * es  # vapor pressure
    vpd = es - e

    return vpd
```
The data was sampled to a 0.25 x 0.25 deg spatial resolution using linear interpolation.

### Surface Soil Moisture (SSM)
The SSM data used for inference is from NASA's GLDAS mission. A summary of the daily GLDAS2.2 products used can be 
found at https://disc.gsfc.nasa.gov/datasets/GLDAS_CLSM025_DA1_D_2.2/summary. The files are hosted at
https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM025_DA1_D.2.2/. The band used is the 'SoilMoist_S_tavg' 
band. In order to sample to a monthly temporal resolution, the sum of each daily SSM value was used for each month.
