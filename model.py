import tensorflow as tf
import numpy as np
import scipy as sp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from json_parser_train import parse_feats
from load import load

test, train, gt_test, gt_train = parse_feats()

X_train = train[0][:,:]
Y_train = gt_train[0,:]
X_test = test[0][0:np.size(test[0][:,:],0)-1,:]
Y_test = gt_test[0,:]

model = Sequential()

model.add(Dense(128, input_dim=66, activation='relu')) 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

"""input1 = keras.layers.Input(shape=(12,))
input2 = keras.layers.Input(shape=(12,))
input3 = keras.layers.Input(shape=(2,))
input4 = keras.layers.Input(shape=(14,))
input5 = keras.layers.Input(shape=(14,))
input6 = keras.layers.Input(shape=(12,))

x1 = keras.layers.Dense(200,activation='relu')(input1)
#d1 = keras.layers.Dropout(0.5)(x1)
x2 = keras.layers.Dense(200,activation='relu')(input2)
#d2 = keras.layers.Dropout(0.5)(x2)
x3 = keras.layers.Dense(200,activation='relu')(input3)
#d3 = keras.layers.Dropout(0.5)(x3)
x4 = keras.layers.Dense(200,activation='relu')(input4)
#d4 = keras.layers.Dropout(0.5)(x4)
x5 = keras.layers.Dense(200,activation='relu')(input5)
#d5 = keras.layers.Dropout(0.5)(x5)
x6 = keras.layers.Dense(200,activation='relu')(input6)
#d6 = keras.layers.Dropout(0.5)(x6)

y1 = keras.layers.Dense(200,activation='relu')(x1)
d21 = keras.layers.Dropout(0.7)(y1)
y2 = keras.layers.Dense(200,activation='relu')(x2)
d22 = keras.layers.Dropout(0.7)(y2)
y3 = keras.layers.Dense(200,activation='relu')(x3)
d23 = keras.layers.Dropout(0.7)(y3)
y4 = keras.layers.Dense(200,activation='relu')(x4)
d24 = keras.layers.Dropout(0.7)(y4)
y5 = keras.layers.Dense(200,activation='relu')(x5)
d25 = keras.layers.Dropout(0.7)(y5)
y6 = keras.layers.Dense(200,activation='relu')(x6)
d26 = keras.layers.Dropout(0.7)(y6)

z1 = keras.layers.Dense(200,activation='relu')(d21)
d31 = keras.layers.Dropout(0.7)(z1)
out1 = keras.layers.Dense(3, activation='softmax')(d31)
z2 = keras.layers.Dense(200,activation='relu')(d22)
d32 = keras.layers.Dropout(0.7)(z2)
out2 = keras.layers.Dense(3, activation='softmax')(d32)
z3 = keras.layers.Dense(200,activation='relu')(d23)
d33 = keras.layers.Dropout(0.7)(z3)
out3 = keras.layers.Dense(3, activation='softmax')(d33)
z4 = keras.layers.Dense(200,activation='relu')(d24)
d34 = keras.layers.Dropout(0.7)(z4)
out4 = keras.layers.Dense(3, activation='softmax')(d34)
z5 = keras.layers.Dense(200,activation='relu')(d25)
d35 = keras.layers.Dropout(0.7)(z5)
out5 = keras.layers.Dense(3, activation='softmax')(d35)
z6 = keras.layers.Dense(200,activation='relu')(d26)
d36 = keras.layers.Dropout(0.7)(z6)
out6 = keras.layers.Dense(3, activation='softmax')(d36)

concat = keras.layers.Concatenate()([out1,out2,out3,out4,out5,out6])

fusion = keras.layers.Dense(18,activation='relu')(concat)
fusion2 = keras.layers.Dense(6,activation='relu')(fusion)

out = keras.layers.Dense(3, activation='softmax')(fusion2)
model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)"""

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""model.fit([X_train[:,0:12], X_train[:,12:24], X_train[:,24:26], X_train[:,26:40], X_train[:,40:54], X_train[:,54:66]], np_utils.to_categorical(Y_train), 
         batch_size=128, nb_epoch=14,validation_data=([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66]], np_utils.to_categorical(Y_test)),verbose=2)"""
model.fit(X_train, np_utils.to_categorical(Y_train), 
         batch_size=128, nb_epoch=14,validation_data=(X_test,np_utils.to_categorical(Y_test)),verbose=2)