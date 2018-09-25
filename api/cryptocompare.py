# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 22:53:28 2018
Derived from: https://medium.com/@galea/cryptocompare-api-quick-start-guide-ca4430a484d4
Modified for use by: Marc A. Ochsner, @marc0ch

Implementation of Cryptocompare's web API for simplified use of Python for time-series investing data.
"""

import requests
import datetime
import pandas as pd
#import matplotlib.pyplot as plt

class cryptocompare:
    
    def price(self, symbol, comparison_symbols=['USD'], exchange=''):
        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}'\
                .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()
        return data
    
    def histo(self, f, t='usd', limit=100, timespan='day', aggregate='100'):
        url = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&aggregate={}'\
                .format(timespan, f.upper(), t.upper(), limit, aggregate)
        page = requests.get(url) 
        print(page)
        data = page.json()['Data']
        print(data)
        df = pd.DataFrame(data)
        print(df)
#        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df
    
    def coin_list(self):
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data
