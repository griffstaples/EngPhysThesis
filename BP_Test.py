import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data
filename = 'Net_Data.txt'
net_data = np.genfromtxt(filename)

#AvgU = np.genfromtxt('Average_Unshuffled.txt')
#Avg = np.genfromtxt('Average.txt')

#AvgU_ans = AvgU[:,-1]
#AvgU = AvgU[:,1:-2]

#Avg_ans = Avg[:,-1]
#Avg = Avg[:,1:-2]
#delete_list = np.linspace(51,55,5)
#delete_list.astype(int)
# net_data = np.delete(net_data,delete_list,axis=0)
# delete_list = np.linspace(63,70,8)
# delete_list.astype(int)
# net_data = np.delete(net_data,delete_list,axis=0)
# delete_list = np.linspace(91,116,20)
# delete_list.astype(int)
# net_data = np.delete(net_data,delete_list,axis=0)
# delete_list = np.linspace(155,202,58)
# delete_list.astype(int)
# net_data = np.delete(net_data,delete_list,axis=0)
answers = net_data[:,-2]
# split = 400
# answersT = answers[:split]
# answersV = answers[split:]
patient = net_data[:,0].copy()
inputs = net_data[:,1:-2]
#inputs = np.delete(inputs,(8),axis =1)
#inputs = np.column_stack((net_data,np.ones(len(net_data[:,0]))))
#age is necessary (1)
#heart rate is also important(4)
#get rid of (7)
#keep (8)
# valid_set = inputs[split:,:]
# inputs = inputs[:split,:]






#define constants
cols = len(inputs[0,:])
#cols = 1
rows = len(inputs[:,0])
#rows = len(inputs)

n_epochs = 300



def mae_std(y_true, y_pred):
    return keras.backend.std(y_true-y_pred)

#build model
model = keras.models.Sequential()
optimizer = keras.optimizers.SGD(lr = 0.005,momentum =0.001, clipnorm = 1.0)
#optimizer = keras.optimizers.Adamax()
initializer = keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)
model.add(keras.layers.Dense(cols+5,activation = tf.nn.sigmoid,name = 'first',
                             kernel_initializer=initializer,
                             input_shape = [cols]))
model.add(keras.layers.Dense(cols,activation = tf.nn.elu,name = 'second',
                              kernel_initializer=initializer))
model.add(keras.layers.Dense(1, activation = 'linear'))

#def ind_error(answers,)

model.compile(optimizer=optimizer,
              loss = 'mean_squared_error',
              metrics = ['mean_absolute_error','mean_squared_error',mae_std])

#plot error descent
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

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
#
# plt.figure(1)
# plt.xlabel('Epoch')
# plt.ylabel('Mean Square Error [$MPG^2$]')
# plt.plot(hist['epoch'], hist['mean_squared_error'],
#          label='Train Error')
# plt.plot(hist['epoch'], hist['val_mean_squared_error'],
#          label='Val Error')
# plt.ylim(0,2000)
# plt.legend()

#guesses = model.predict(valid_set)
#error = guesses-answersV

print(hist['val_mean_absolute_error'])
x = model.predict(inputs)

# for c,i in enumerate(x):
#     print(i-AvgU_ans[c])

answers = np.reshape(answers,(653,1))
mae = np.abs(x-answers)
print(mae)
print(np.mean(mae))
print(np.std(mae))
##print(hist['val_mae_std'])
#
#bins = np.linspace(-5,35,12)
#plt.hist(mae,bins=bins,histtype='barstacked')
#plt.show()

#first_layer_weights = model.layers[0].get_weights()[0]
#print(np.mean(np.abs(first_layer_weights),axis=1))

#second_layer_weights = model.layers[1].get_weights()[0]

#print(first_layer_weights)
#print(second_layer_weights)