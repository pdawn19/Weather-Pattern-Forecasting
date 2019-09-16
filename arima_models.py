from statsmodels.tsa.arima_model import ARIMA,ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import numpy as np

#ARIMA/ARMA returns a 2D array as predictions 
#while SARIMA returns mean predictions
def model_arima(train_data, test_data):

    # Basic ARIMA Model fitting
    model = ARIMA(train_data, order=(4, 1, 0))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary().tables[1])

    # works for v0.6.1 only
    # model_fit.plot_diagnostics(figsize=(16, 8))
    # plt.tight_layout()
    # plt.show()

    # plot residual errors
    residuals = model_fit.resid
    plt.subplot(2,2,1)
    plt.plot(residuals, color='r')
    plt.title('ARIMA Model Residuals')
    plt.subplot(2,2,2)
    sns.distplot(residuals, hist=True, kde=True,
                 bins=int(180 / 5), color='blue',
                 hist_kws={'edgecolor': 'black'})
    plt.subplot(2,2,3)
    probplot(residuals, dist='norm', fit=True, plot=plt)
    plt.subplot(2,2,4)
    plot_acf(residuals)
    plt.show()

    # Forecast
    step = 10
    predictions = model_fit.forecast(steps=step)
    # print(predictions, test_data[0:step])
    plt.plot(predictions[0], color='r', label='Mean Prediction')
    # plt.plot(predictions[1], color='g')
    plt.plot(predictions[2], color='b', label='Variance')
    plt.plot(test_data[0:step], color='k', label='Actual')
    plt.legend(loc='best')
    plt.title('ARIMA predictions')
    plt.show()
#    print(np.shape(predictions))

    print('ARIMA RMSE: ', mean_squared_error(predictions[0], test_data[0:step]))
    
 
def model_arma(train_data,test_data):
    model = ARMA(train_data, order= (1,1))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary().tables[1])
    # Forecast
    step = 10
    predictions = model_fit.forecast(steps=step)
#    print(np.shape(predictions))
    print('ARMA RMSE: ', mean_squared_error(predictions[0], test_data[0:step]))


def model_sarima(train_data,test_data):
    model = SARIMAX(train_data, order = (1,1,1), seasonal_order = (1,1,1,1))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary().tables[1])
    # Forecast
    step = 10
    predictions = model_fit.forecast(steps=step)
    print('SARIMA RMSE: ', mean_squared_error(predictions, test_data[0:step]))
    print(np.shape(predictions))
    
    
def model_sarimax(train_data,test_data,train_data1,test_data1):
    model = SARIMAX(train_data1, exog = train_data, order = (1,1,1), seasonal_order = (1,1,1,1))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary().tables[1])
    # Forecast
    step = 10
    predictions = model_fit.forecast(steps=step, exog = test_data[0:step].reshape((step,1)))
    print('SARIMAX RMSE: ', mean_squared_error(predictions, test_data1[0:step]))
    print(np.shape(predictions)) 
    
def model_var(train_data,test_data,train_data1,test_data1):
    x = train_data1.reshape((372,1))
    x1 = train_data.reshape((372,1))
    lis = np.concatenate((x,x1), axis = 1)
    print(np.shape(lis))
    #forecast
    model = VAR(endog = lis)
    model_fit = model.fit()
    print(model_fit.summary())
    predictions = model_fit.forecast(model_fit.y, steps=10)
    print('VAR RMSE: ', mean_squared_error(predictions[:,0], test_data1[0:10]))
    
def model_varmax(train_data,test_data,train_data1,test_data1):
    x = train_data1.reshape((372,1))
    x1 = train_data.reshape((372,1))
    lis = np.concatenate((x,x1), axis = 1)
    print(np.shape(lis))
    #forecast
    model = VARMAX(lis, order=(1,1))
    model_fit = model.fit(disp = -1)
    print(model_fit.summary().tables[1])
    predictions = model_fit.forecast(steps=10)
    print('VARMAX RMSE: ', mean_squared_error(predictions[:,0], test_data1[0:10]))
    
    
