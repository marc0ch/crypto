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

from statsmodels.sandbox.regression.predstd import wls_prediction_std
from pandas.core import datetools #used f
from pandas import Series, DataFrame, Panel
from scipy import stats
from itertools import product
from datetime import datetime

from pydoc import help
from scipy.stats.stats import pearsonr

quandl.ApiConfig.api_key = "vkS5iE4CbLvky5oneS9F"

df_gold = quandl.get("WGC/GOLD_DAILY_USD",start_date="2012-01-01",end_date="2017-05-31")
#np.polyfit(df_gold.Date,df_gold.Value,deg=3)
df_gold = df_gold.reset_index()
#np.polyfit(df_gold.Date,df_gold.Value,deg=3)
df_gold.Timestamp = pd.to_datetime(df_gold.Date, unit='d')
#np.polyfit(df_gold.Timestamp,df_gold.Value,deg=2)
df_gold.index = df_gold.Timestamp
df_gold = df_gold.resample('D').mean()
######################
nsample = df_gold.shape[0]
x = np.linspace(0, 10, nsample)
X = np.column_stack((x,x**2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
X = sm.add_constant(X)
y = np.dot(X, beta) + e

# fit data
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# NEW TEST
print('Parameters: ', results.params)
print('R2: ', results.rsquared)


# OLS non-linear curve but linear in parameter
#~~~~~~~~~~~~~~~~~~~~~
#nsample = 50
#sig = 0.5
#x = np.linspace(0, 20, nsample)
#X = np.column_stack((x, np.sin(x), (x-5)**2, np.ones(nsample)))
#beta = [0.5, 0.5, -0.02, 5.]
#
#y_true = np.dot(X, beta)
#y = y_true + sig * np.random.normal(size=nsample)
##
#res = sm.OLS(y, X).fit()
#print(res.summary())
##
#print('Parameters: ', res.params)
#print('Standard errors: ', res.bse)
#print('Predicted values: ', res.predict())
##
#prstd, iv_l, iv_u = wls_prediction_std(res)
##
#fig, ax = plt.subplots(figsize=(8,6))
##
#ax.plot(x, y, 'o', label="data")
#ax.plot(x, y_true, 'b-', label="True")
#ax.plot(x, res.fittedvalues, 'r--.', label="OLS")
#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
#ax.legend(loc='best');
#~~~~~~~~~~~~~~~~~~~~

##############################
#np.polyfit(df_gold.index,df_gold.Value,deg=2)
#df_gold['i'] = range(0,df_gold.shape[0]-1)

#est = sm.OLS(df_gold['Value'], df_gold[['e', 'varA', 'meanM', 'varM', 'covAM']]).fit()
#df_gold.rsquared_adj()
#est.summary()

# numpy.polyfit with degree 'd' fits a linear regression with the mean function E(y|x) = p_d * x**d + p_{d-1} * x **(d-1) + ... + p_1 * x + p_0
#df_gold.Timestamp = pd.to_datetime(df_gold.Timestamp, unit='d')
#df_gold.index = df_gold.Timestamp
#df_gold = df_gold.resample('D').mean()
#print(list(df_gold)[0])
#print(list(df_gold)[1])
print(type(df_gold)," DTYPES:")
print(df_gold.dtypes)
print(list(df_gold))
print("  shape")
print(df_gold.shape)
#print(df_gold.shape[0])
#for i in range(0,df_gold.shape[0]):
#    print(df_gold.Value[i],df_gold['dtype:])
    
#print(df_gold.Value[17000])
#print(df_gold.Date)
df_btc = pd.read_csv('btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv')
df_snp = pd.read_csv('SP500.csv')
df_snp['Price'] = df_snp.SP500.apply(pd.to_numeric, errors='coerce')
print(df_snp)
#print(df_snp)

print('----------------------------------------------------------')
print(list(df_snp),df_snp.dtypes,df_snp.shape)
#print(df_snp.open.dtype,"<--open")
#print(df_snp.date.dtype,"<--date")
#df_btc.head()
df_btc = df_btc.reset_index()
df_snp = df_snp.reset_index()
#df_btc['Date']
df_btc.Timestamp = pd.to_datetime(df_btc['Timestamp'], unit='s')
df_snp.Timestamp = pd.to_datetime(df_snp['DATE'])
df_btc.index = df_btc.Timestamp
df_snp.index = df_snp.Timestamp
#df_snp.price = df_snp.SP500.
df_btc = df_btc.resample('D').mean()
#print(df_btc)
print(type(df_btc)," DTYPES:")
print(df_btc.dtypes)
print(list(df_btc))
print('  shape')
print(df_btc.shape)
print(df_btc.index)
#print(df_btc)

fig = plt.figure(figsize=[15,7])
plt.suptitle('Price ($)', fontsize=22)

#RENAME
new_col_names = []
for i in df_btc.columns.values:
    s = i.replace("(","")
    s = s.replace(")","")
    new_col_names.append(s)
df_btc.columns = new_col_names


# NORMAL X,Y PLOT
plt.plot([math.exp(y) for y in df_btc.Weighted_Price], '-', label='BTC by Days')
#plt.plot(df_gold.Value, '-', label='Gold by Days',log='true')
#plt.plot(df_snp.Price,'-',label='S&P By Days',log='true')
plt.yscale=('log')
plt.show()

# Resampling to monthly frequency
df_btc_M = df_btc.resample('M').mean()
df_gold_M = df_gold.resample('M').mean()
df_snp_M = df_snp.resample('M').mean()

# Resampling to annual frequency
df_btc_Y = df_btc.resample('A-DEC').mean()
df_gold_Y = df_gold.resample('A-DEC').mean()
df_snp_Y = df_snp.resample('A-DEC').mean()


# Resampling to quarterly frequency
df_btc_Q = df_btc.resample('Q-DEC').mean()
df_gold_Q = df_gold.resample('Q-DEC').mean()
df_snp_Q = df_snp.resample('Q-DEC').mean()

# Resampling to monthly frequency
df_btc_M = df_btc.resample('M').mean()
df_gold_M = df_gold.resample('M').mean()
df_snp_M = df_snp.resample('M').mean()

# Resampling to annual frequency
df_btc_Y = df_btc.resample('A-DEC').mean()
df_gold_Y = df_gold.resample('A-DEC').mean()
df_snp_Y = df_snp.resample('A-DEC').mean()

# Resampling to quarterly frequency
df_btc_Q = df_btc.resample('Q-DEC').mean()
df_gold_Q = df_gold.resample('Q-DEC').mean()
df_snp_Q = df_snp.resample('Q-DEC').mean()

plt.plot(df_btc_M.Weighted_Price, '-', label='BTC by Month')
plt.plot(df_gold_M.Value, '-', label='Gold by Month')
plt.plot(df_snp_M.Price, '-', label='S&P by Month')
plt.legend()
plt.show()
plt.plot(df_btc_Q.Weighted_Price, '-', label='BTC by Q')    
plt.plot(df_gold_Q.Value, '-', label='Gold by Q')
plt.plot(df_snp_Q.Price, '-', label='Gold by Q')
plt.legend()
plt.show()
plt.plot(df_btc_Y.Weighted_Price, '-', label='BTC by Year')
plt.plot(df_gold_Y.Value, '-', label='Gold by Year')
plt.plot(df_snp_Y.Price, '-', label='S&P by Year')
plt.legend()
plt.show()


fig = plt.figure(figsize=[15, 7])
plt.suptitle('Bitcoin Volume ($)', fontsize=22)
plt.plot(df_btc_Y.Volume_BTC,'r-',label='Volume (BTC)', alpha=0.1)
plt.plot(df_btc_Y.Volume_Currency,'y-',label='Volume (Currency)', alpha=0.1)

plt.legend()

# TREND ANALYSIS -> SM (statsmodels.api)
plt.figure(figsize=[15,7])
sm.tsa.seasonal_decompose(df_btc_M.Weighted_Price).plot()
sm.tsa.seasonal_decompose(df_gold_M.Value).plot()
sm.tsa.seasonal_decompose(df_snp_M.Price).plot()
#print("Dickey–Fuller test (BTC): p=%f" % sm.tsa.stattools.adfuller(df_btc_M.Weighted_Price)[1])
#print("Dickey–Fuller test (GOLD): p=%f" % sm.tsa.stattools.adfuller(df_gold_M.Value)[1])

# TREND ANALYSIS -> SCIPY
#print("PASSSED sm.tsa.seasonal_decompose(df_btc_M.Weighted_Price).plot()")
plt.figure(figsize=[15,7])
#plt.figure()
# Box-Cox Transformations df_snp_M.Price
df_btc_M['Weighted_Price_box'], lmbda = stats.boxcox(df_btc_M.Weighted_Price)
df_gold_M['Weighted_Price_box'], lmbda = stats.boxcox(df_gold_M.Value)
df_snp_M['Weighted_Price_box'], lmbda = stats.boxcox(df_snp_M.Price)
print("Dickey–Fuller test (BTC): p=%f" % sm.tsa.stattools.adfuller(df_btc_M.Weighted_Price)[1])
print("Dickey–Fuller test (GOLD): p=%f" % sm.tsa.stattools.adfuller(df_gold_M.Value)[1])
print("Dickey–Fuller test (SNP): p=%f" % sm.tsa.stattools.adfuller(df_snp_M.Price)[1])

# Seasonal differentiation -> BOX COX TRANSFORMATION
df_btc_M['prices_box_diff'] = df_btc_M.Weighted_Price_box - df_btc_M.Weighted_Price_box.shift(12)
df_gold_M['prices_box_diff'] = df_gold_M.Weighted_Price_box - df_gold_M.Weighted_Price_box.shift(12)
df_snp_M['prices_box_diff'] = df_snp_M.Weighted_Price_box - df_snp_M.Weighted_Price_box.shift(12)
print("Dickey–Fuller test (BTC): p=%f" % sm.tsa.stattools.adfuller(df_btc_M.prices_box_diff[12:])[1])
print("Dickey–Fuller test (GOLD): p=%f" % sm.tsa.stattools.adfuller(df_gold_M.prices_box_diff[12:])[1])
print("Dickey–Fuller test (SNP): p=%f" % sm.tsa.stattools.adfuller(df_snp_M.prices_box_diff[12:])[1])

# Regular differentiation
df_btc_M['prices_box_diff2'] = df_btc_M.prices_box_diff - df_btc_M.prices_box_diff.shift(1)
df_gold_M['prices_box_diff2'] = df_gold_M.prices_box_diff - df_gold_M.prices_box_diff.shift(1)
df_snp_M['prices_box_diff2'] = df_snp_M.prices_box_diff - df_snp_M.prices_box_diff.shift(1)
plt.figure(figsize=(15,7))

# STL-decomposition
sm.tsa.seasonal_decompose(df_btc_M.prices_box_diff2[13:]).plot()  
print("Dickey–Fuller test (BTC): p=%f" % sm.tsa.stattools.adfuller(df_btc_M.prices_box_diff2[13:])[1])
sm.tsa.seasonal_decompose(df_gold_M.prices_box_diff2[13:]).plot()
print("Dickey–Fuller test (GOLD): p=%f" % sm.tsa.stattools.adfuller(df_gold_M.prices_box_diff2[13:])[1])
sm.tsa.seasonal_decompose(df_snp_M.prices_box_diff2[13:]).plot()
print("Dickey–Fuller test (SNP): p=%f" % sm.tsa.stattools.adfuller(df_snp_M.prices_box_diff2[13:])[1])


# Initial Approx of parameters using Autocorrelation / Partial Autocorrelation Plots
plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_btc_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_btc_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()

plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_gold_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_gold_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()

plt.figure(figsize=(15,7))
ax = plt.subplot(211)
sm.graphics.tsa.plot_acf(df_snp_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
ax = plt.subplot(212)
sm.graphics.tsa.plot_pacf(df_snp_M.prices_box_diff2[13:].values.squeeze(), lags=48, ax=ax)
plt.tight_layout()

#common = df_btc.merge(df_gold,on=['index','index']) #inner
#df_btc[~df_btc.isin(df_gold)].dropna()

#np.corrcoef(df_gold['Value'], df_btc['Open'])
#
#mpl.style.use('ggplot')
#
#plt.scatter(df_gold['Value'], df_btc['Open'])
#plt.show()
#print(df1.dtypes)
#print(df2.dtypes)
# TODO: Convert BTC TO DAILY, WEEKLY
# TODO: GENERATE WEEKEND DATES
#df2 = df2.where()

#list_dfs = [df1,df2]
#def save_xls(list_dfs, xls_path):
#    writer = pd.ExcelWriter(xls_path)
#    i=0
#    for n, df in enumerate(list_dfs):
#        df.to_excel(writer,df.__name__)
#        i=i+1
#    writer.save()
#save_xls(list_dfs,'dash_data.xlsx')