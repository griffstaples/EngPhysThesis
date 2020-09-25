#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Griffin Staples
Date Created: Thu Feb 21 2019
License:
The MIT License (MIT)
Copyright (c) 2019 Griffin Staples

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
filepath = './Data/Net_Data.txt'
net_data = np.genfromtxt(filepath)

#slice data
answers = net_data[:,-1] #systolic blood pressure
answers = net_data[:,-2] #diastolic blood pressure
patient = net_data[:,0].copy()
inputs = net_data[:,1:-2]


#define constants
cols = len(inputs[0,:])
rows = len(inputs[:,0])
n_epochs = 300


def mae_std(y_true, y_pred):
    return keras.backend.std(y_true-y_pred)

#build model
model = keras.models.Sequential()
optimizer = keras.optimizers.SGD(lr = 0.005,momentum =0.001, clipnorm = 1.0)
initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
model.add(keras.layers.Dense(cols+5,activation = tf.nn.sigmoid,name = 'first',
                             kernel_initializer=initializer,
                             input_shape = [cols]))
model.add(keras.layers.Dense(cols,activation = tf.nn.elu,name = 'second',
                              kernel_initializer=initializer))
model.add(keras.layers.Dense(1, activation = 'linear'))

#compile model with error metrics
model.compile(optimizer=optimizer,
              loss = 'mean_squared_error',
              metrics = ['mean_absolute_error','mean_squared_error',mae_std])

#plot error descent
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#fit model
history = model.fit(
  inputs, answers,
  epochs=n_epochs, verbose=0,
  validation_split = 0.3, callbacks = [PrintDot()])


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plt.figure(0)
plt.xlabel('Epoch')
plt.ylabel('Mean Abs Error [mmHg]')
plt.plot(hist['epoch'], hist['mean_absolute_error'],
         label='Train Error')
plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
         label='Val Error')
plt.legend()
plt.ylim(0,40)
plt.show()

x = model.predict(inputs)

mae = np.abs(x-answers)
print("MAE: ", np.mean(mae))
print("STD on MAE: ", np.std(mae))