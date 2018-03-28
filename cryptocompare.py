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
    
    def price(symbol, comparison_symbols=['USD'], exchange=''):
        url = 'https://min-api.cryptocompare.com/data/price?fsym={}&tsyms={}'\
                .format(symbol.upper(), ','.join(comparison_symbols).upper())
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()
        return data
    
    def daily_price_historical(symbol, comparison_symbol, all_data=True, limit=1, aggregate=1, exchange=''):
        url = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&aggregate={}'\
                .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        if all_data:
            url += '&allData=true'
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df
    
    
    def hourly_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
        url = 'https://min-api.cryptocompare.com/data/histohour?fsym={}&tsym={}&limit={}&aggregate={}'\
                .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df
    
    def minute_price_historical(symbol, comparison_symbol, limit, aggregate, exchange=''):
        url = 'https://min-api.cryptocompare.com/data/histominute?fsym={}&tsym={}&limit={}&aggregate={}'\
                .format(symbol.upper(), comparison_symbol.upper(), limit, aggregate)
        if exchange:
            url += '&e={}'.format(exchange)
        page = requests.get(url)
        data = page.json()['Data']
        df = pd.DataFrame(data)
        df['timestamp'] = [datetime.datetime.fromtimestamp(d) for d in df.time]
        return df
    
    def coin_list():
        url = 'https://www.cryptocompare.com/api/data/coinlist/'
        page = requests.get(url)
        data = page.json()['Data']
        return data
