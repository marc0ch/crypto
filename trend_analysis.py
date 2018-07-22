# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:41:45 2017

@author: Owner
"""

import quandl
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import math
import requests

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from pandas.core import datetools #used f
from pandas import Series, DataFrame, Panel
from scipy import stats
from itertools import product
from datetime import datetime

from pydoc import help
from scipy.stats.stats import pearsonr

import crypto_compare as cc

#btc_h = api_.cryptocompare.hourly_price_historical('btc','usd')
btc_m = cc.cryptocompare.minute_price_historical('btc','usd')
print(btc_m.shape)

btc_m = btc_m.reset_index()
btc_m.Timestamp = pd.to_datetime(btc_m['timestamp'], unit='d')
btc_m.index = btc_m.Timestamp

btc_D = btc_m.resample('D').mean()
btc_M = btc_m.resample('M').mean()
btc_Y = btc_m.resample('Y').mean()


btc_M['Weighted_Price_box'], lmbda = stats.boxcox(btc_M.Weighted_Price)
