# 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15650700.svg)](https://doi.org/10.5281/zenodo.15650700)

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

# Getting started

## Creating the environment

After cloning this repo to a write-accessible directory, use the requirements.txt or environment.yml file to create the 
virtual environment for running the code. From within the fdeo directory, run one of following sets of code in the
terminal depending on whether you would like to use a virtual environment or conda environment

### Virtual Environment
```commandline
python -m venv fdeo_env

// macOS / Linux 
source fdeo_env/bin/activate

// Windows
fdeo_env\Scripts\activate

pip install -r requirements.txt
```

### Conda
```commandline
conda env create -f environment.yml
conda activate fdeo_env
```

## Running the code


### Credentials
First, make sure the environment is activated. The main script is at fdeo/fdeo.py. Because this script will make calls 
to the NASA earthdata API, you must supply your urs.earthdata.nasa.gov login credentials. These can either be passed in
as command line arguments, or set as environment variables (recommended). To set your credentials as env variables, run
the following commands in the shell you will be running the code from

macOS / Linux
```bash
export FDEO_USER=<your_earthdata_username>
export FDEO_PWD=<your_earthdata_password>
```

To set these permanently, add this command to your ~/.bashrc or ~/.bash_profile

Windows
```powershell
$env:FDEO_USER=<your_earthdata_username>
$env:FDEO_PWD=<your_earthdata_password>
```

To set these permanently add them in your system settings


### Calling the executable
Call the fdeo.py script from the root project directory (or add the project directory to the PYTHONPATH environment
variable) like 

```bash
python fdeo/fdeo.py 
```

If your credentials are not set as environment variables, pass them in as command line arguments

```bash
python fdeo/fdeo.py -u <earthdata_username> -p <earthdata_password> 
```

Make sure to escape any special characters, like '$'

This will gather API data from the previous month in the data store based on the current time. Generally, data 
will be available up to the previous month. Running this script in August 2023, for example, will gather API data for 
July 2023, and generate predictions for September 2023.

If you would like a custom start and end date for the API data retrieval, you can specify those with flags like so
```bash
python fdeo/fdeo.py -u <earthdata_username> -p <earthdata_password> --start_date 2023-02 --end_date 2023-05 
```

This will generate predictions for April - July 2023

### Output

Whenever the script is run, the coefficients for the model will be looked for at data/model_coefficients.pkl. If this 
file does not exist, the model will be trained again and this file will be written out for use in subsequent calls.
Plots of the fire observation data and the predictions over the 2003 - 2013 training data range will also be looked for
in data/training_data_observation_results and data/training_data_prediction_results. If these do not exist they will be
created so that one can analyze the performance of the model over this period for which truth data exists. 

The new prediction data will be written to data/prediction_results/<prediction_start_date>_<prediction_end_date>.
For example, if one were to call 
```bash
python fdeo/fdeo.py -u <earthdata_username> -p <earthdata_password> --start_date 2023-02 --end_date 2023-05 
```
Then a directory with the results will be created at data/prediction_results/2023.04_2023.07.
This directory will contain two files called prediction_categorical.tif and prediction_probability.tif which contain a
band for each month of their respective prediction metrics (categorical or probabilistic). There are also two folders 
called categorical_plots and probability plots which contain images of the prediction plots for each month.

### How to cite this tool

Verleye, E., Fanguna, L., & Amaral, C. (2025). Fire Danger from Earth Observations (FDEO) Python package. Zenodo. https://doi.org/10.5281/zenodo.15650700

