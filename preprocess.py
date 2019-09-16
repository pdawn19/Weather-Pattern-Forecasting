import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = np.mean(rolling_window(timeseries, window=12), -1)
    rolstd = np.std(rolling_window(timeseries, window=12), -1)

    # Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    if dftest[0] < dftest[4]['5%']:
        print('Series is Stationary')


def pre(data):

    # Autocorrelation plot of the time series
#    autocorrelation_plot(data)
#    plt.title("Autocorr with confidence interval")
#    plt.show()
    # q value from QACF, p value from ACF
    plot_acf(data)
    plt.title("Autocorr with lag markers")
    plt.show()
    plot_pacf(data)
    plt.title("Partial Autocorr with lag markers")
    plt.show()

    # Check stationary
    test_stationarity(data)

    # Decompose into Components
    result = seasonal_decompose(data, model='additive', freq=48)
    result.plot()
    plt.show()

    # Estimating and Eliminating Trend
    ln_data = (data)
    moving_avg = np.mean(rolling_window(ln_data, 5), -1)
    plt.plot(ln_data, color='blue', label='Data')
    plt.plot(moving_avg, color='red', label='Moving Avg')
    plt.legend(loc='best')
    plt.title('Estimated Trend')
    plt.show()
