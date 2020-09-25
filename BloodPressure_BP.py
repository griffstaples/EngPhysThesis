from sklearn.neural_network import MLPRegressor
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

filename = 'Net_Data.txt'
net_data = np.genfromtxt(filename)
answers = net_data[:,-2]
patient = net_data[:,0].copy()


data_set = net_data[:,1:-2]
cols = len(data_set[0,:])
rows = len(data_set[:,0])




print(data_set.shape)

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(cols, activation=tf.nn.relu, input_shape=[cols]),
    keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


model = build_model()
model.summary()

print(model.predict(data_set[0,:].reshape(1,8)))

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 10

history = model.fit(
  data_set, answers,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

print()
print(hist)


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(0)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()

    plt.figure(1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.show()


plot_history(history)


print(model.predict(data_set).flatten())




#
# a_0 = tf.placeholder(tf.float32,[1,cols])
# y = tf.placeholder(tf.float32, [1,2])
#
# hidden_nodes = cols
#
# w_1 = tf.Variable(tf.truncated_normal([cols, hidden_nodes]))
# b_1 = tf.Variable(tf.truncated_normal([1, hidden_nodes]))
# w_2 = tf.Variable(tf.truncated_normal([hidden_nodes, 2]))
# b_2 = tf.Variable(tf.truncated_normal([1, 2]))
#
# def sigma(x):
#     return tf.divide(tf.constant(1.0),
#                   tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))
#
# def sigmaprime(x):
#     return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))
#
#
#
# z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
# a_1 = sigma(z_1)
# z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
# a_2 = z_2
#
# diff = tf.subtract(a_2,y)
#
# #calculate change in weights
# d_z_2 = tf.multiply(diff, 1.0)
# d_b_2 = d_z_2
# d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)
#
# d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
# d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
# d_b_1 = d_z_1
# d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)
#
#
# #learning rate
# eta = tf.constant(0.5)
#
# step = [
#     tf.assign(w_1,
#             tf.subtract(w_1, tf.multiply(eta, d_w_1)))
#   , tf.assign(b_1,
#             tf.subtract(b_1, tf.multiply(eta,
#                                tf.reduce_mean(d_b_1, axis=[0]))))
#   , tf.assign(w_2,
#             tf.subtract(w_2, tf.multiply(eta, d_w_2)))
#   , tf.assign(b_2,
#             tf.subtract(b_2, tf.multiply(eta,
#                                tf.reduce_mean(d_b_2, axis=[0]))))
# ]
#
# #cost = tf.multiply(diff,diff)
# #step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#
# acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
# acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
# err = tf.subtract(a_2,y)
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
#
#
#
# for i in range(n_epochs):
#     for j in range(rows):
#         sess.run(step,feed_dict = {a_0: data_set[i,:].reshape((1,8)),
#                                    y: answers[i,:].reshape((1,2))})
#     error = sess.run(err,feed_dict={a_0: data_set[i,:].reshape((1,8)),
#                                    y: answers[i,:].reshape((1,2))})
#     print(error)


print('Success')
