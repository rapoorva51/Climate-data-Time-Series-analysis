#!/usr/bin/env python
# coding: utf-8

# # Climate data Time Series analysis

# In[1]:


import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')

import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


from math import sqrt

import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
import seaborn as sns

from random import random

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


# In[2]:


df = pd.read_csv(r'C:\Users\HP\Downloads\climate_data (2).csv', usecols=[0,1,2,3,4,5,6,7,8,9], engine='python',header = None)
df


# In[3]:


df.dropna( axis=0,how='any')


# In[4]:


df.isnull().any()


# In[5]:


df.isnull().sum()


# In[6]:


df.rename(columns={'Average temperature (Â°F)' : 'avgtemp'},inplace=True)
df.rename(columns={'Average humidity (%)' : 'avghumid'},inplace=True)
df.rename(columns={'Average dewpoint (Â°F)' : 'avgdew'},inplace=True)
df.rename(columns={'Average barometer (in)' : 'augbaro'},inplace=True)
df.rename(columns={'Average windspeed (mph)' : 'augwind'},inplace=True)
df.rename(columns={'Average gustspeed (mph)' : 'avggust'},inplace=True)
df.rename(columns={'Average direction (Â°deg)' : 'avgdir'},inplace=True)
df.rename(columns={'Rainfall for month (in)' : 'rfm'},inplace=True)
df.rename(columns={'Rainfall for year (in)' : 'rfy'},inplace=True)


# In[7]:


df.columns = ['Date','avgtemp','avghumid','avgdew','augbaro','augwind','avggust','avgdir','rfm','rfy']


# In[8]:


df.describe()


# In[9]:


df.shape


# In[10]:


df.dtypes


# In[11]:


# Data Preprocessing and Visualization


# In[12]:


df.columns


# In[13]:


cols = ['avgtemp', 'avghumid', 'avgdew', 'augbaro', 'augwind', 'avggust', 'avgdir', 'rfm', 'rfy']
df[cols] = df[cols].apply(pd.to_numeric,errors='coerce')


# In[14]:


import pandas as pd
df['Date']= pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df.info()


# In[15]:


y = df.set_index('Date')


# In[16]:


y.index


# In[18]:


y.drop(y.head(1).index,inplace=True)


# In[19]:


y.isnull().any()


# In[20]:


y.isnull().sum()


# In[21]:


y.plot(figsize=(15, 6))
plt.show()


# In[25]:



from random import randrange
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = [i+randrange(10) for i in range(1,100)]
result = seasonal_decompose(series, model='additive', period=1)
result.plot()
pyplot.show()


# In[31]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf


pyplot.figure()
pyplot.subplot(211)
plot_acf(y.avgtemp, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.avgtemp, ax=pyplot.gca(), lags = 30)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(y.avghumid, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.avghumid, ax=pyplot.gca(), lags = 30)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(y.avgdew, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.avgdew, ax=pyplot.gca(), lags = 30)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(y.augbaro, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.augbaro, ax=pyplot.gca(), lags = 30)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(y.augwind, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.augwind, ax=pyplot.gca(), lags = 30)
pyplot.show()

pyplot.figure()
pyplot.subplot(211)
plot_acf(y.avggust, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.avggust, ax=pyplot.gca(), lags = 30)
pyplot.show()


pyplot.figure()
pyplot.subplot(211)
plot_acf(y.avgdir, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.avgdir, ax=pyplot.gca(), lags = 30)
pyplot.show()


pyplot.figure()
pyplot.subplot(211)
plot_acf(y.rfm, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.rfm, ax=pyplot.gca(), lags = 30)
pyplot.show()


pyplot.figure()
pyplot.subplot(211)
plot_acf(y.rfy, ax=pyplot.gca(), lags = 30)
pyplot.subplot(212)
plot_pacf(y.rfy, ax=pyplot.gca(), lags = 30)
pyplot.show()


# Reviewing plots of the density of observations can provide further insight into the structure of the data:
# 
# The distribution is not perfectly Gaussian (normal distribution).
# The distribution is left shifted.
# Transformations might be useful prior to modelling.

# In[33]:


from statsmodels.tsa.stattools import adfuller


# In[36]:


#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Temperature:')
dftest = adfuller(y.avgtemp, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Humidity:')
dftest = adfuller(y.avghumid, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg DewPoint:')
dftest = adfuller(y.avgdew, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Windspeed:')
dftest = adfuller(y.augwind, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Barometer:')
dftest = adfuller(y.augbaro, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Gust speed:')
dftest = adfuller(y.avggust, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg rainfall of Month:')
dftest = adfuller(y.rfm, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Rainfall of Year:')
dftest = adfuller(y.rfy, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test Avg Direction:')
dftest = adfuller(y.avgdir, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


# In[37]:


ts_log = np.log(y)
plt.plot(ts_log)


# In[40]:


from pandas import Series
from matplotlib import pyplot
pyplot.figure(1)
pyplot.subplot(211)
y.avgtemp.hist()
pyplot.subplot(212)
y.avgtemp.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(2)
pyplot.subplot(211)
y.avghumid.hist()
pyplot.subplot(212)
y.avghumid.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(3)
pyplot.subplot(211)
y.avghumid.hist()
pyplot.subplot(212)
y.avghumid.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(4)
pyplot.subplot(211)
y.avgdew.hist()
pyplot.subplot(212)
y.avgdew.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(4)
pyplot.subplot(211)
y.augbaro.hist()
pyplot.subplot(212)
y.augbaro.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(4)
pyplot.subplot(211)
y.augwind.hist()
pyplot.subplot(212)
y.augwind.plot(kind='kde')
pyplot.show()


from pandas import Series
from matplotlib import pyplot
pyplot.figure(7)
pyplot.subplot(211)
y.avggust.hist()
pyplot.subplot(212)
y.avggust.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(8)
pyplot.subplot(211)
y.avgdir.hist()
pyplot.subplot(212)
y.avgdir.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(9)
pyplot.subplot(211)
y.rfm.hist()
pyplot.subplot(212)
y.rfm.plot(kind='kde')
pyplot.show()

from pandas import Series
from matplotlib import pyplot
pyplot.figure(10)
pyplot.subplot(211)
y.rfy.hist()
pyplot.subplot(212)
y.rfy.plot(kind='kde')
pyplot.show()


# Autoregression (AR)
# The autoregression (AR) method models the next step in the sequence as a linear function of the observations at prior time steps.
# Number of AR (Auto-Regressive) terms (p): p is the parameter associated with the auto-regressive aspect of the model, which incorporates past values i.e lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).

# In[50]:


ts_log_diff = ts_log.avgtemp - ts_log.avgtemp.shift()
plt.plot(ts_log_diff)


# In[52]:


from statsmodels.tsa.ar_model import AR
from random import random


# In[54]:


# fit model
model = AR(ts_log_diff)
model_fit = model.fit()


# In[55]:


plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))
plt.show()


# In[56]:


predictions_ARIMA_diff = pd.Series(model_fit.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())


# In[57]:


predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())


# In[58]:


predictions_ARIMA_log = pd.Series(ts_log.avgtemp.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[59]:


predictions_ARIMA = np.exp(predictions_ARIMA_log)


# In[60]:


plt.plot(y.avgtemp)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(np.nansum((predictions_ARIMA-y.avgtemp)**2)/len(y.avgtemp)))


# Forecast quality scoring metrics
# R squared
# Mean Absolute Error
# Median Absolute Error
# Mean Squared Error
# Mean Squared Logarithmic Error
# Mean Absolute Percentage Error

# In[61]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error, mean_squared_log_error


# In[62]:


r2_score(y.avgtemp, predictions_ARIMA)


# In[63]:


mean_absolute_error(y.avgtemp, predictions_ARIMA)


# In[65]:


median_absolute_error(y.avgtemp, predictions_ARIMA)


# In[66]:


mean_squared_error(y.avgtemp, predictions_ARIMA)


# In[69]:


def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y.avgtemp, predictions_ARIMA)


# Function to evaluate forecast using above metrics:

# In[72]:


def evaluate_forecast(y,pred):
    results = pd.DataFrame({'r2_score':r2_score(y, pred),
                           }, index=[0])
    results['mean_absolute_error'] = mean_absolute_error(y, pred)
    results['median_absolute_error'] = median_absolute_error(y, pred)
    results['mse'] = mean_squared_error(y, pred)
    results['mape'] = mean_absolute_percentage_error(y, pred)
    results['rmse'] = np.sqrt(results['mse'])
    return results


# In[73]:


evaluate_forecast(y.avgtemp, predictions_ARIMA)


# Moving Average (MA)
# Number of MA (Moving Average) terms (q): q is size of the moving average part window of the model i.e. lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.

# In[74]:


# MA example
from statsmodels.tsa.arima_model import ARMA
from random import random

# fit model
model = ARMA(ts_log_diff, order=(0, 1))
model_fit = model.fit(disp=False)


# In[75]:


model_fit.summary()


# In[76]:


plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))


# Autoregressive Moving Average (ARMA)
# Number of AR (Auto-Regressive) terms (p): p is the parameter associated with the auto-regressive aspect of the model, which incorporates past values i.e lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).

# In[77]:


# ARMA example
from statsmodels.tsa.arima_model import ARMA
from random import random

# fit model
model = ARMA(ts_log_diff, order=(2, 1))
model_fit = model.fit(disp=False)


# In[78]:


model_fit.summary()


# In[79]:


plt.plot(ts_log_diff)
plt.plot(model_fit.fittedvalues, color='red')
plt.title('RSS: %.4f'% np.nansum((model_fit.fittedvalues-ts_log_diff)**2))


# Autoregressive Integrated Moving Average (ARIMA)
# In an ARIMA model there are 3 parameters that are used to help model the major aspects of a times series: seasonality, trend, and noise. These parameters are labeled p,d,and q.
# 
# Number of AR (Auto-Regressive) terms (p): p is the parameter associated with the auto-regressive aspect of the model, which incorporates past values i.e lags of dependent variable. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5).
# Number of Differences (d): d is the parameter associated with the integrated part of the model, which effects the amount of differencing to apply to a time series.
# Number of MA (Moving Average) terms (q): q is size of the moving average part window of the model i.e. lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value.

# In[80]:


ts = y.avgtemp - y.avgtemp.shift()
ts.dropna(inplace=True)


# In[81]:


pyplot.figure()
pyplot.subplot(211)
plot_acf(ts, ax=pyplot.gca(),lags=30)
pyplot.subplot(212)
plot_pacf(ts, ax=pyplot.gca(),lags=30)
pyplot.show()


# In[82]:


#divide into train and validation set
train = y[:int(0.75*(len(y)))]
valid = y[int(0.75*(len(y))):]

#plotting the data
train['avgtemp'].plot()
valid['avgtemp'].plot()


# In[100]:


train.head()


# In[101]:


train_prophet = pd.DataFrame()
train_prophet['ds'] = train.index
train_prophet['y'] = train.avgtemp.values


# In[102]:


train_prophet.head()


# In[103]:


from fbprophet import Prophet

#instantiate Prophet with only yearly seasonality as our data is monthly 
model = Prophet( yearly_seasonality=True, seasonality_mode = 'multiplicative')
model.fit(train_prophet) #fit the model with your dataframe


# In[104]:


# predict for five months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 36, freq = 'MS') 
future.tail()


# In[107]:


forecast = model.predict()
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

#plot the predictions for validation set
plt.plot(y.avgtemp, label='Train')
#plt.plot(valid, label='Valid')
plt.plot(forecast, label='Prediction')
plt.show()


# In[108]:


forecast.columns


# In[109]:


# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[110]:


fig = model.plot(forecast)
#plot the predictions for validation set

plt.plot(valid, label='Valid', color = 'red', linewidth = 2)

plt.show()


# In[111]:


model.plot_components(forecast)


# In[112]:


y_prophet = pd.DataFrame()
y_prophet['ds'] = y.index
y_prophet['y'] = y.avgtemp.values


# In[113]:


y_prophet = y_prophet.set_index('ds')
forecast_prophet = forecast.set_index('ds')


# In[117]:


evaluate_forecast(y_prophet.y[1:500], forecast_prophet.yhat[1:500])


# In[ ]:




