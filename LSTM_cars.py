# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 04:42:15 2022

@author: Sebastian Gumula

Time series prediction of shampoo sales dataset with LSTM model
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_absolute_percentage_error
import numpy

def mape(actual, pred): 
    actual, pred = numpy.array(actual), numpy.array(pred)
    return numpy.mean(numpy.abs((actual - pred) / actual)) * 100

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


loop = 5
rmse_list, mape_list = list(),list()
for LOOP in range(loop):
    #Wczytanie datasetu
    load_df = pd.read_csv("cars.csv", header =0, index_col= 0)
    raw_values = load_df.values
    #Różnicowanie datasetu
    diff_df = load_df.diff(1,0)
    diff_df.fillna(0, inplace=True)
    
    #Transformacja do uczenia nadzorowanego
    
    columns = [diff_df.shift(i) for i in range(1, 2)]
    columns.append(diff_df)
    supervised = pd.concat(columns, axis=1)
    supervised.fillna(0, inplace=True)
    #Podział na train, test data
    supervised = supervised.values
    train, test = supervised[:72],supervised[72:]
    
    #Skalowanie danych trenujących
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    scaled_train = scaler.transform(train)
    
    #Skalowanie danych testowych
    scaler = scaler.fit(test)
    scaled_test = scaler.transform(test)
    
    #Dopasowanie modelu
    lstm_model = fit_lstm(scaled_train,1,500,2)
    print(lstm_model)
    train_reshaped = scaled_train[:, 0].reshape(len(scaled_train), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    print(lstm_model.summary())
    
    #Walidacja modelu
    predictions = list()
    for i in range(len(scaled_test)):
    	# make one-step forecast
    	X, y = scaled_test[i, 0:-1], scaled_test[i, -1]
    	yhat = forecast_lstm(lstm_model, 1, X)
    	# invert scaling
    	yhat = invert_scale(scaler, X, yhat)
    	# invert differencing
    	yhat = inverse_difference(raw_values, yhat, len(scaled_test)+1-i)
    	# store forecast
    	predictions.append(yhat)
    	expected = raw_values[len(train) + i]
    	#print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
    
    
    # report performance
    rmse = sqrt(mean_squared_error(raw_values[-36:], predictions))
    mape_val = mape(raw_values[-36:],predictions)
    print('Test RMSE ' + str(LOOP) + ': %.3f' % rmse)
    print(mape(raw_values[-36:],predictions))
    rmse_list.append(rmse)
    mape_list.append(mape_val)
    
    # line plot of observed vs predicted
    plt.plot(raw_values[-36:])
    plt.plot(predictions)
    plt.xlabel("Miesiąc")
    plt.ylabel("Ilość")
    plt.show()
    
print(sum(rmse_list))
print(sum(mape_list))