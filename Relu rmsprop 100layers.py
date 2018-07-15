
#model with reLu activation function and RMSprop optimizer

#import all of the functions and classes we intend to use
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
from matplotlib import pyplot


#load the dataset
import csv
import pandas as pd
import os as os

dataframe=pd.read_csv('sample2.csv',sep=',')
dataset = dataframe.values
dataset = dataset.astype('float32')

#shape of data
dataset.shape


# fix random seed for reproducibility
numpy.random.seed(7)



# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

scaler = MinMaxScaler(feature_range=(0, 1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)



# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:24]
        dataX.append(a)
        dataY.append(dataset[i + look_back,24])
    return numpy.array(dataX), numpy.array(dataY)



# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test,look_back)
trainX.shape




# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[2]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[2]))
print(trainX.shape, trainY.shape, testX.shape, testY.shape)




# create and fit the LSTM network
model = Sequential()

model.add(LSTM(50, input_shape = (look_back,24),return_sequences=True))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=True))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=False))

model.add(LSTM(
    50,
    return_sequences=True))


model.add(Dense(
    output_dim=1))
model.add(Activation('relu'))
start = time.time()
model.compile(loss='mse', optimizer='rmsprop', metrics=['mse', 'accuracy'])
print ('compilation time : ', time.time() - start)




history=model.fit(
    trainX,
    trainY,
    batch_size=128,
    nb_epoch=100,
    validation_split=0.05)


# do some predictioms


yhat= model.predict(testX)
yhat
yhat.shape

# reshape the data
test_X = testX.reshape((testX.shape[0], testX.shape[2]))
test_X
test_X.shape


# concatenate and y predictions
from numpy import concatenate
inv_yhat = concatenate((test_X,yhat), axis=1)
inv_yhat


# transform back the data
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat
inv_yhat.shape


# get the y value of the transformed data
inv_yhat =inv_yhat[:,24]
inv_yhat

# reshape testY to get reshaped original y values on test set
test_y = testY.reshape((len(testY), 1))
test_y
test_y.shape

# concatenate the original value of y in first column of test set without the last column

inv_y = concatenate((test_X[:, 0:],test_y), axis=1)
inv_y


# transform the set above


inv_y = scaler.inverse_transform(inv_y)
inv_y


# take the last column


inv_y = inv_y[:,24]
inv_y
inv_y.shape


# check that inv_y.shape and inv_yhat.shape has the same shape

inv_yhat.shape
inv_yhat

# plot the loss and validation loss with each epoch 

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
plt.ylabel('value')
plt.xlabel('epoch')
pyplot.legend()
pyplot.show()


# plot the predicted value and the observed value for each observation
# or import values of predicted and observed and plot on R 

from matplotlib import pyplot
pyplot.plot(inv_yhat, label='predicted value')
pyplot.plot(inv_y, label='observed value')
plt.ylabel('value')
plt.xlabel('observations')
pyplot.legend()
pyplot.show()



