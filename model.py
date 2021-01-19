import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Dense
from keras.regularizers import l2

my_model = Sequential()

my_model.add(LSTM(units = 64,input_shape = (33,1),return_sequences=True,kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001), recurrent_regularizer=l2(0.00001)))
my_model.add(LSTM(units = 128, return_sequences=True,kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001), recurrent_regularizer=l2(0.00001)))

my_model.add(Dense(100, kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001)))
my_model.add(Dense(10, kernel_regularizer=l2(0.00001),bias_regularizer=l2(0.00001)))
my_model.add(Dense(1))
opt = tf.keras.optimizers.Adam(learning_rate=0.00025)#changed from 0.001
my_model.compile(loss = 'mse',optimizer = opt, metrics = ['mean_squared_error'])
my_model.summary()