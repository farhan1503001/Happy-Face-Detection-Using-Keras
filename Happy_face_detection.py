# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:08:30 2018

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
data_train=h5py.File('train_happy.h5',mode="r")
data_test=h5py.File('test_happy.h5',mode='r')
for key in data_train.keys():
    print(key)
x_train=np.array(data_train["train_set_x"].value)
y_train=np.array(data_train["train_set_y"].value)
y_train=y_train.reshape(600,1)
x_test=np.array(data_test["test_set_x"].value)
y_test=np.array(data_test["test_set_y"].value)
y_test=y_test.reshape(150,1)
#Now seeing some data
plt.imshow(x_train[23])
plt.show()
import keras as k
x_train=k.utils.normalize(x_train,axis=-1)
model=k.models.Sequential()
model.add(k.layers.Conv2D(32,(3,3),strides=(1,1),activation='relu'))
model.add(k.layers.MaxPool2D())
model.add(k.layers.Conv2D(64,(3,3),strides=(1,1),activation='relu'))
model.add(k.layers.MaxPool2D())
model.add(k.layers.Flatten())
model.add(k.layers.Dense(512,activation='relu'))
model.add(k.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,batch_size=12)
x_test=k.utils.normalize(x_test,axis=-1)
val_loss,val_accuracy=model.evaluate(x_test,y_test)
model.save("happydetect.h5")

model.summary()
#Predicting the result
predictions=model.predict_classes(x_test)
plt.imshow(x_test[0])
plt.show()

