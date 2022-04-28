import sys

sys.path.append(
    "/uio/kant/geo-metos-u1/franzihe/Documents/Python/globalsnow/eosc-nordic-climate-demonstrator"
)

import xarray as xr

# xr.set_options(display_style="html")
import intake
import cftime
import cartopy.crs as ccrs
import cartopy as cy
import matplotlib.pyplot as plt

import xesmf as xe
from glob import glob
import pandas as pd
import numpy as np

from cmcrameri import cm
from scipy.stats import linregress
import dask.array as da

import geocat.comp as gc
from datetime import datetime, timedelta

import h5py

from scipy.optimize import curve_fit

import functions as fct
