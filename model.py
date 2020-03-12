# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 03:29:49 2020

@author: kwamekert
"""

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#importing datasets
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test,y_test) = cifar10.load_data()

#transforming data between 0 and 1

x_train = x_train/255

x_test = x_test/255

#lableing data 
from tensorflow.keras.utils import to_categorical

y_cat_train = to_categorical(y_train,10)

y_cat_test = to_categorical(y_test,10)


#start convolution
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


model = Sequential()

#convolutional layer
model.add(Conv2D(filters = 32, kernel_size=(4,4),input_shape=(32,32,3), activation ='relu'))

#pooling layer
model.add(MaxPool2D(pool_size=(2,2)))


#more convolution

#convolutional layer
model.add(Conv2D(filters = 32, kernel_size=(4,4),input_shape=(32,32,3), activation ='relu'))

#pooling layer
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])



model.summary()


from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2)

#model.fit(x_train,y_cat_train,epochs=15,validation_data=(x_test,y_cat_test),callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)























