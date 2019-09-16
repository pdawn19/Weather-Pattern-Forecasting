# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 20:59:37 2018

@author: dawnp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import pre
from arima_models import *

if __name__ == '__main__':
    # Load Data
    data = pd.read_csv("weather-data/110-tavg-all-1-1980-2018.csv", names = ['d','t','a'])
    temp = data.iloc[1:, 1].astype(float)
    rain_data = pd.read_csv("weather-data/110-pcp-all-1-1980-2018.csv", names = ['d', 't', 'a'] )
    rainfall = rain_data.iloc[1:, 1].astype(float)
    # pressure = pressure.iloc[2:, 1].astype(float)
    # temp = pd.read_csv("weather-data/temperature.csv", header=None, low_memory=False)
    # temp = temp.iloc[2:, 1].astype(float)
    date = data.iloc[1:, 0]
#    date, time = [], []
#    for item in datetime:
#        a, b = item.split(' ')
#        date.append(a)
#        time.append(b)

    # Data Pre-processing - Fill Missing Values
    temp = temp.fillna(method='ffill', axis=0)
    rainfall = rainfall.fillna(method='ffill', axis=0)
    # temp = temp.fillna(method='ffill', axis=0)

    # Typecasting and Data Splitting
    temp = np.asarray(temp)
    train_temp, test_temp = temp[0:372], temp[372:]
    rainfall = np.asarray(rainfall)
    train_rain, test_rain = rainfall[0:372], rainfall[372:]
    train_date, test_date = date[0:372], date[372:]
#    print(np.shape(test_temp), len(train_date))
    print(np.shape(test_rain), len(train_date))

    # Plotting training data-temperature
#    plt.plot(train_date, train_temp)
#    plt.title("Temperature")
#    plt.xticks(rotation=90)
#    plt.rc('xtick', labelsize=5)
#    plt.autoscale(enable=True, axis='x', tight=True)
#    plt.tight_layout()
#    plt.show()
    
    # Plotting training data-rainfall
    plt.plot(train_date, train_rain)
    plt.title("Rainfall")
    plt.xticks(rotation=90)
    plt.rc('xtick', labelsize=5)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.show()

    # Preprocessing Data
    pre(train_temp)
    pre(train_rain)

    # Fitting Univariate Data-temperature
    model_arima(train_temp, test_temp)
    model_arma(train_temp, test_temp)
    model_sarima(train_temp,test_temp)
    
#    # Fitting Univariate Data-rainfall
#    model_arima(train_rain, test_rain)
#    model_arma(train_rain, test_rain)
#    model_sarima(train_rain,test_rain)
#    
    

    # Fitting Multivariate Data
#    model_sarimax(train_temp,test_temp,train_rain,test_rain)
#    model_var(train_temp,test_temp,train_rain,test_rain)
#    model_varmax(train_temp,test_temp,train_rain,test_rain)
