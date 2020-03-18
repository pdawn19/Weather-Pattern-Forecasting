import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocess import pre
from arima_models import *

if __name__ == '__main__':
    # Load Data
    data = pd.read_csv("weather-data/humidity.csv", header=None, low_memory=False)
    humidity = data.iloc[25:, 1].astype(float)
    pressure = pd.read_csv("weather-data/pressure.csv", header=None, low_memory=False)
    pressure = pressure.iloc[2:, 1].astype(float)
    temp = pd.read_csv("weather-data/temperature.csv", header=None, low_memory=False)
    temp = temp.iloc[2:, 1].astype(float)
    datetime = data.iloc[25:, 0]
    date, time = [], []
    for item in datetime:
        a, b = item.split(' ')
        date.append(a)
        time.append(b)

    # Data Pre-processing - Fill Missing Values
    humidity = humidity.fillna(method='ffill', axis=0)
    pressure = pressure.fillna(method='ffill', axis=0)
    temp = temp.fillna(method='ffill', axis=0)

    # Typecasting and Data Splitting
    humidity = np.asarray(humidity)
    train_humidity, test_humidity = humidity[0:36000:168], humidity[36000::168]
    train_date, test_date = date[0:36000:168], date[36000::168]
    print(np.shape(test_humidity), len(train_date))

    # Plotting training data
    plt.plot(train_date, train_humidity)
    plt.title("Humidity")
    plt.xticks(rotation=90)
    plt.rc('xtick', labelsize=5)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.tight_layout()
    plt.show()

    # Preprocessing Data
    pre(train_humidity)

    # Fitting Data
    model_arima(train_humidity, test_humidity)
