Weather-Pattern-Forecasting

Location guided forecast on local/global scale using Univariate/Multivariate Fore-casting Models like ARIMA, ARMA, SARIMAX,VARMAX
BigDataWeatherAnalysis using Apache Spark

This project was implemented using Python 3.6 and requires the folllowing libraries to be installed as they're not found on the core Python package.

pyspark
statsmodels
pandas
numpy
matplotlib

The Time Series Analysis was performed in WeatherTimeSeriesAnalysis.py. The analysis was performed for the forecast of temperature. The dataset contains hourly data. It was resampled for daily and monthly forecasts and the mean of the respective day/month was used for the resampled data. The time sries analysis was perfomed using Autoregressive Integrated Moving Averages(ARIMA). The predictions and forecasts were performed for both daily and monthly samples.
