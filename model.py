import tensorflow as tf
import numpy as np
import scipy as sp
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from json_parser_train import parse_feats

test, train, gt_test, gt_train = parse_feats()

X_train = train[0][:,:]
Y_train = gt_train[0,:]
X_test = test[0][0:np.size(test[0][:,:],0)-1,:]
Y_test = gt_test[0,:]

model = Sequential()

model.add(Dense(128, input_dim=66, activation='relu')) 
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, 
         batch_size=32, nb_epoch=5,validation_data=(X_test, Y_test),verbose=2)