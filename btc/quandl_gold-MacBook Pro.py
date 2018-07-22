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
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime

df1 = quandl.get("WGC/GOLD_DAILY_USD")
df2 = pd.read_csv('btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv')

# TODO: Convert BTC TO DAILY, WEEKLY
# TODO: GENERATE WEEKEND DATES
df2 = df2.where()

list_dfs = [df1,df2]
def save_xls(list_dfs, xls_path):
    writer = pd.ExcelWriter(xls_path)
    i=0
    for n, df in enumerate(list_dfs):
        df.to_excel(writer,df.__name__)
        i=i+1
    writer.save()
save_xls(list_dfs,'dash_data.xlsx')