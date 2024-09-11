 # -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:29:54 2022

@author: Sebastian Gumula
"""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from pmdarima import auto_arima
import numpy as np
import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("cars.csv", header=0, index_col=0)

df_values = df.values     
  
diff = list()

for i in range(1, len(diff)):
    value = df_values[i] - df_values[i-1]
    diff.append(value)

#Splitting data into train and test sets
size = 72
train, test = df_values[0:size], df_values[size:len(df_values)]
"""
arimamodel= ARIMA(train, order=(4,2,1))
arimamodel_fit = arimamodel.fit()
print(arimamodel_fit.summary())
"""
autoarima_model=auto_arima(train, max_p=10,max_q=10,max_d=10,test="adf",trace=True, stationary=True)
print(autoarima_model.summary())

history = [x for x in train]
predictions = list()


# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(3,0,2))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t] 
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
    
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

# plot forecasts against actual outcomes

plt.plot(test,label="Dane Testowe")
plt.plot(predictions, label= "Predykcje", color='red')

plt.xlabel("Miesiąc")
plt.ylabel("Ilość")
plt.title("Predykcje modelu ARIMA(3,0,2)")

plt.show()

mape = mean_absolute_percentage_error(test, predictions)
print(mape)
print(len(df))
print(len(test))
print(len(predictions))

