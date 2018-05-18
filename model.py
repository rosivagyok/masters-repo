import tensorflow as tf
import numpy as np
import scipy as sp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
#from json_parser_train import parse_feats
from load import load

test, train, gt_test, gt_train = load()

X_train = train[1][:,:]
#X_train = np.reshape(X_train,(np.int(X_train.shape[0]/8), 8, X_train.shape[1]))
#Y_train = gt_train[1,0::8]
Y_train = gt_train[1,:]
X_test = test[1][0:test[1].shape[0]-1,:]
#X_test = np.reshape(X_test[0:len(X_test)-7,:],(np.int(X_test.shape[0]/8), 8, X_test.shape[1]))
#Y_test = gt_test[1,0:np.size(gt_test[1])-7:8]
Y_test = gt_test[1,:]

shape = 10

model = late_dnn_nod()
history, pred = evaluate(model,depth_label=False)

def evaluate(model,depth_label):

    if depth_label==True:
        history = model.fit([X_train[:,0:12], X_train[:,12:24], X_train[:,24:26], X_train[:,26:40], X_train[:,40:54], X_train[:,54:66], X_train[:,66:78]], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=64, nb_epoch=75,validation_data=([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66], X_test[:,66:78]], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66], X_test[:,66:78]], batch_size=32, verbose=2, steps=None)
    else:
        history = model.fit([X_train[:,0:12], X_train[:,12:24], X_train[:,24:26], X_train[:,26:40], X_train[:,40:54], X_train[:,54:66]], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=64, nb_epoch=75,validation_data=([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66]], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66]], batch_size=32, verbose=2, steps=None)
    return history, pred

def late_DNN(depth_label=False):
    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(12,))

    x1 = keras.layers.Dense(128,activation='relu')(input1)
    d1 = keras.layers.Dropout(0.5)(x1)

    x2 = keras.layers.Dense(128,activation='relu')(input2)
    d2 = keras.layers.Dropout(0.5)(x2)

    x3 = keras.layers.Dense(128,activation='relu')(input3)
    d3 = keras.layers.Dropout(0.5)(x3)

    x4 = keras.layers.Dense(128,activation='relu')(input4)
    d4 = keras.layers.Dropout(0.5)(x4)

    x5 = keras.layers.Dense(128,activation='relu')(input5)
    d5 = keras.layers.Dropout(0.5)(x5)

    x6 = keras.layers.Dense(128,activation='relu')(input6)
    d6 = keras.layers.Dropout(0.5)(x6)

    if depth_label==True:
        x7 = keras.layers.Dense(128,activation='relu')(input7)
        d7 = keras.layers.Dropout(0.5)(x7)

    y1 = keras.layers.Dense(128,activation='relu')(d1)
    d21 = keras.layers.Dropout(0.5)(y1)

    y2 = keras.layers.Dense(128,activation='relu')(d2)
    d22 = keras.layers.Dropout(0.5)(y2)

    y3 = keras.layers.Dense(128,activation='relu')(d3)
    d23 = keras.layers.Dropout(0.5)(y3)

    y4 = keras.layers.Dense(128,activation='relu')(d4)
    d24 = keras.layers.Dropout(0.5)(y4)

    y5 = keras.layers.Dense(128,activation='relu')(d5)
    d25 = keras.layers.Dropout(0.5)(y5)

    y6 = keras.layers.Dense(128,activation='relu')(d6)
    d26 = keras.layers.Dropout(0.5)(y6)

    if depth_label==True:
        y7 = keras.layers.Dense(128,activation='relu')(d7)
        d27 = keras.layers.Dropout(0.5)(y7)

    z1 = keras.layers.Dense(128,activation='relu')(d21)
    d31 = keras.layers.Dropout(0.5)(z1)
    out1 = keras.layers.Dense(3, activation='softmax')(d31)

    z2 = keras.layers.Dense(128,activation='relu')(d22)
    d32 = keras.layers.Dropout(0.5)(z2)
    out2 = keras.layers.Dense(3, activation='softmax')(d32)

    z3 = keras.layers.Dense(128,activation='relu')(d23)
    d33 = keras.layers.Dropout(0.5)(z3)
    out3 = keras.layers.Dense(3, activation='softmax')(d33)

    z4 = keras.layers.Dense(128,activation='relu')(d24)
    d34 = keras.layers.Dropout(0.5)(z4)
    out4 = keras.layers.Dense(3, activation='softmax')(d34)

    z5 = keras.layers.Dense(128,activation='relu')(d25)
    d35 = keras.layers.Dropout(0.5)(z5)
    out5 = keras.layers.Dense(3, activation='softmax')(d35)

    z6 = keras.layers.Dense(128,activation='relu')(d26)
    d36 = keras.layers.Dropout(0.5)(z6)
    out6 = keras.layers.Dense(3, activation='softmax')(d36)

    if depth_label==True:
        z7 = keras.layers.Dense(128,activation='relu')(d27)
        d37 = keras.layers.Dropout(0.5)(z7)
        out7 = keras.layers.Dense(3, activation='softmax')(d37)

    if depth_label==True:
        concat = keras.layers.Concatenate()([out1,out2,out3,out4,out5,out6,out7])
    else:
        concat = keras.layers.Concatenate()([out1,out2,out3,out4,out5,out6])

    fusion = keras.layers.Dense(18,activation='relu')(concat)
    fusion2 = keras.layers.Dense(6,activation='relu')(fusion)

    out = keras.layers.Dense(3, activation='softmax')(fusion2)

    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    
    return model

def early_DNN(depth_label=False):

    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(12,))

    x1 = keras.layers.Dense(128,activation='relu')(input1)
    d1 = keras.layers.Dropout(0.5)(x1)

    x2 = keras.layers.Dense(128,activation='relu')(input2)
    d2 = keras.layers.Dropout(0.5)(x2)

    x3 = keras.layers.Dense(128,activation='relu')(input3)
    d3 = keras.layers.Dropout(0.5)(x3)

    x4 = keras.layers.Dense(128,activation='relu')(input4)
    d4 = keras.layers.Dropout(0.5)(x4)

    x5 = keras.layers.Dense(128,activation='relu')(input5)
    d5 = keras.layers.Dropout(0.5)(x5)

    x6 = keras.layers.Dense(128,activation='relu')(input6)
    d6 = keras.layers.Dropout(0.5)(x6)

    if depth_label==True:
        x7 = keras.layers.Dense(128,activation='relu')(input7)
        d7 = keras.layers.Dropout(0.5)(x7)

    y1 = keras.layers.Dense(128,activation='relu')(d1)
    d21 = keras.layers.Dropout(0.5)(y1)

    y2 = keras.layers.Dense(128,activation='relu')(d2)
    d22 = keras.layers.Dropout(0.5)(y2)

    y3 = keras.layers.Dense(128,activation='relu')(d3)
    d23 = keras.layers.Dropout(0.5)(y3)

    y4 = keras.layers.Dense(128,activation='relu')(d4)
    d24 = keras.layers.Dropout(0.5)(y4)

    y5 = keras.layers.Dense(128,activation='relu')(d5)
    d25 = keras.layers.Dropout(0.5)(y5)

    y6 = keras.layers.Dense(128,activation='relu')(d6)
    d26 = keras.layers.Dropout(0.5)(y6)

    if depth_label==True:
        y7 = keras.layers.Dense(128,activation='relu')(d7)
        d27 = keras.layers.Dropout(0.5)(y7)

    z1 = keras.layers.Dense(128,activation='relu')(d21)
    d31 = keras.layers.Dropout(0.5)(z1)

    z2 = keras.layers.Dense(128,activation='relu')(d22)
    d32 = keras.layers.Dropout(0.5)(z2)

    z3 = keras.layers.Dense(128,activation='relu')(d23)
    d33 = keras.layers.Dropout(0.5)(z3)

    z4 = keras.layers.Dense(128,activation='relu')(d24)
    d34 = keras.layers.Dropout(0.5)(z4)

    z5 = keras.layers.Dense(128,activation='relu')(d25)
    d35 = keras.layers.Dropout(0.5)(z5)

    z6 = keras.layers.Dense(128,activation='relu')(d26)
    d36 = keras.layers.Dropout(0.5)(z6)

    if depth_label==True:
        z7 = keras.layers.Dense(128,activation='relu')(d27)
        d37 = keras.layers.Dropout(0.5)(z7)
        fusion = keras.layers.Dense(64,activation='relu')(d31,d32,d33,d34,d35,d36,d37)
    else:
        fusion = keras.layers.Dense(64,activation='relu')(d31,d32,d33,d34,d35,d36)

    out = keras.layers.Dense(3, activation='softmax')(fusion)

    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model

def late_LSTM(depth_label=False):

    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(12,))

    x1 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input1)
    x2 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input2)
    x3 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 2),
                       dropout=0.5)(input3)
    x4 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 14),
                       dropout=0.5)(input4)
    x5 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 14),
                       dropout=0.5)(input5)
    x6 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input6)
    if depth_label==True:
        x7 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input7)

    y1 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x1)
    d1 = keras.layers.Dense(128,activation='relu')(y1)
    out1 = keras.layers.Dense(3, activation='softmax')(d1)

    y2 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x2)
    d2 = keras.layers.Dense(128,activation='relu')(y1)
    out2 = keras.layers.Dense(3, activation='softmax')(d2)

    y3 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 2),
                       dropout=0.5)(x3)
    d3 = keras.layers.Dense(128,activation='relu')(y1)
    out3 = keras.layers.Dense(3, activation='softmax')(d3)

    y4 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 14),
                       dropout=0.5)(x4)
    d4 = keras.layers.Dense(128,activation='relu')(y1)
    out4 = keras.layers.Dense(3, activation='softmax')(d4)

    y5 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 14),
                       dropout=0.5)(x5)
    d5 = keras.layers.Dense(128,activation='relu')(y1)
    out5 = keras.layers.Dense(3, activation='softmax')(d5)

    y6 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x6)
    d6 = keras.layers.Dense(128,activation='relu')(y1)
    out6 = keras.layers.Dense(3, activation='softmax')(d6)

    if depth_label==True:
        y7 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x7)
        d7 = keras.layers.Dense(128,activation='relu')(y7)
        out7 = keras.layers.Dense(3, activation='softmax')(d7)
        concat = keras.layers.Concatenate()([out1,out2,out3,out4,out5,out6,out7])
    else:
        concat = keras.layers.Concatenate()([out1,out2,out3,out4,out5,out6])

    fusion = keras.layers.Dense(18,activation='relu')(concat)
    fusion2 = keras.layers.Dense(6,activation='relu')(fusion)

    out = keras.layers.Dense(3, activation='softmax')(fusion2)
    
    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model

def early_LSTM(depth_label=False):

    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(12,))

    x1 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input1)
    x2 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input2)
    x3 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 2),
                       dropout=0.5)(input3)
    x4 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 14),
                       dropout=0.5)(input4)
    x5 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 14),
                       dropout=0.5)(input5)
    x6 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input6)

    if depth_label==True:
        x7 = keras.layers.LSTM(128,return_sequences=True,
                       input_shape=(shape, 12),
                       dropout=0.5)(input7)

    y1 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x1)
    d1 = keras.layers.Dense(128,activation='relu')(y1)

    y2 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x2)
    d2 = keras.layers.Dense(128,activation='relu')(y1)

    y3 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 2),
                       dropout=0.5)(x3)
    d3 = keras.layers.Dense(128,activation='relu')(y1)

    y4 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 14),
                       dropout=0.5)(x4)
    d4 = keras.layers.Dense(128,activation='relu')(y1)

    y5 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 14),
                       dropout=0.5)(x5)
    d5 = keras.layers.Dense(128,activation='relu')(y1)

    y6 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x6)
    d6 = keras.layers.Dense(128,activation='relu')(y1)

    if depth_label==True:
        y7 = keras.layers.LSTM(128,return_sequences=False,
                       input_shape=(shape, 12),
                       dropout=0.5)(x7)
        d7 = keras.layers.Dense(128,activation='relu')(y7)
        fusion = keras.layers.Dense(64,activation='relu')([d1,d2,d3,d4,d5,d6,d7])
    else:
        fusion = keras.layers.Dense(64,activation='relu')(d1,d2,d3,d34,d5,d6)

    out = keras.layers.Dense(3, activation='softmax')(fusion)


    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model
