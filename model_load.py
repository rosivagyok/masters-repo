import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
from experiments import get_cnf_mat

def late_DNN(depth_label):
    
    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(6,))

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

    out = keras.layers.Dense(3, activation='softmax')(fusion)

    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    
    return model

def early_DNN(depth_label):
    
    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(6,))

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
        fusion_pre = keras.layers.Concatenate()([d31,d32,d33,d34,d35,d36,d37])
    else:
        fusion_pre = keras.layers.Concatenate()([d31,d32,d33,d34,d35,d36])

    fusion = keras.layers.Dense(64,activation='relu')(fusion_pre)
    out = keras.layers.Dense(3, activation='softmax')(fusion)

    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model

def late_LSTM(shape, depth_label):

    input1 = keras.layers.Input(shape=(12,))
    input2 = keras.layers.Input(shape=(12,))
    input3 = keras.layers.Input(shape=(2,))
    input4 = keras.layers.Input(shape=(14,))
    input5 = keras.layers.Input(shape=(14,))
    input6 = keras.layers.Input(shape=(12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(6,))

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

def early_LSTM(shape,depth_label):

    input1 = keras.layers.Input(shape=(shape,12,))
    input2 = keras.layers.Input(shape=(shape,12,))
    input3 = keras.layers.Input(shape=(shape,2,))
    input4 = keras.layers.Input(shape=(shape,14,))
    input5 = keras.layers.Input(shape=(shape,14,))
    input6 = keras.layers.Input(shape=(shape,12,))

    if depth_label==True:
        input7 = keras.layers.Input(shape=(shape,6,))

    x1 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input1)
    x2 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input2)
    x3 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 2),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input3)
    x4 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 14),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input4)
    x5 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 14),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input5)
    x6 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input6)

    if depth_label==True:
        x7 = keras.layers.LSTM(256,return_sequences=True,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(input7)

    y1 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x1)
    d1 = keras.layers.Dense(128,activation='relu')(y1)

    y2 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x2)
    d2 = keras.layers.Dense(128,activation='relu')(y2)

    y3 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 2),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x3)
    d3 = keras.layers.Dense(128,activation='relu')(y3)

    y4 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 14),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x4)
    d4 = keras.layers.Dense(128,activation='relu')(y4)

    y5 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 14),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x5)
    d5 = keras.layers.Dense(128,activation='relu')(y5)

    y6 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x6)
    d6 = keras.layers.Dense(128,activation='relu')(y6)

    if depth_label==True:
        y7 = keras.layers.LSTM(256,return_sequences=False,
                       input_shape=(shape, 12),
                       recurrent_dropout=0.3,
                       dropout=0.5)(x7)
        d7 = keras.layers.Dense(128,activation='relu')(y7)
        fusion_pre = keras.layers.Concatenate(64,activation='relu')([d1,d2,d3,d4,d5,d6,d7])
    else:
        fusion_pre = keras.layers.Concatenate()([d1,d2,d3,d4,d5,d6])

    fusion = keras.layers.Dense(64,activation='relu')(fusion_pre)
    out = keras.layers.Dense(3, activation='softmax')(fusion)


    if depth_label==True:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6, input7], outputs=out)
    else:
        model = keras.models.Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model

def simple_LSTM(shape):
    input1 = keras.layers.Input(shape=(shape,66,))
    input2 = keras.layers.Input(shape=(shape,6,))
    
    concat = keras.layers.Concatenate()([input1,input2])

    x1 = keras.layers.LSTM(256,return_sequences=True,
                            activation='tanh',
                            input_shape=(shape, 72),
                            recurrent_dropout=0.3,
                            dropout=0.5)(concat)

    y1 = keras.layers.LSTM(256,return_sequences=False,
                            activation='tanh',
                            recurrent_dropout=0.3,
                            dropout=0.5)(x1)

    d1 = keras.layers.Dense(64,activation='relu')(y1)
    out = keras.layers.Dense(3, activation='softmax')(d1)

    model = keras.models.Model(inputs=[input1, input2], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer = keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])

    return model

def evaluate(model, X_train, Y_train, X_test, Y_test, X_depth_train, X_depth_test, depth_label):
    
    if depth_label==False:
        history = model.fit([X_train[:,0:12], X_train[:,12:24], X_train[:,24:26], X_train[:,26:40], X_train[:,40:54], X_train[:,54:66], X_depth_train], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=32, nb_epoch=75,validation_data=([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66], X_depth_test], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66], X_depth_test], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)
        #EARLY: 0.7618 0.6864
        #LATE: 0.7374 0.6612
    else:
        history = model.fit([X_train[:,0:12], X_train[:,12:24], X_train[:,24:26], X_train[:,26:40], X_train[:,40:54], X_train[:,54:66]], np_utils.to_categorical(Y_train,num_classes=3), 
                 batch_size=32, nb_epoch=75,validation_data=([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66]], np_utils.to_categorical(Y_test,num_classes=3)),verbose=2)

        pred = model.predict([X_test[:,0:12], X_test[:,12:24], X_test[:,24:26], X_test[:,26:40], X_test[:,40:54], X_test[:,54:66]], batch_size=32, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(Y_test,class_pred)
        #EARLY: 0.7478 0.6644
        #LATE: 0.7352 0.6506
    return history, pred, cnf_matrix

def evaluate_lstm(model, train, gt_train, test, 
                  gt_test, depth_train, depth_test, depth_label, simple):
    
    if simple == True:
        history = model.fit([train, depth_train], np_utils.to_categorical(gt_train,num_classes=3), 
                 batch_size=16, nb_epoch=1,validation_data=([test, depth_test], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2,shuffle=False)
        pred = model.predict([test, depth_test], batch_size=16, verbose=2, steps=None)
        class_pred = pred.argmax(axis=-1)
        cnf_matrix = get_cnf_mat(gt_test,class_pred)
    else:
        if depth_label==True:
            history = model.fit([train[:,:,0:12], train[:,:,12:24], train[:,:,24:26], train[:,:,26:40], train[:,:,40:54], train[:,:,54:66], depth_train], np_utils.to_categorical(gt_train,num_classes=3), 
                     batch_size=32, nb_epoch=75,validation_data=([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66], depth_test], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2)
        
            pred = model.predict([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66], test[:,:,66:72], depth_test], batch_size=32, verbose=2, steps=None)
            class_pred = pred.argmax(axis=-1)
            cnf_matrix = get_cnf_mat(gt_test,class_pred)
        else:
            history = model.fit([train[:,:,0:12], train[:,:,12:24], train[:,:,24:26], train[:,:,26:40], train[:,:,40:54], train[:,:,54:66]], np_utils.to_categorical(gt_train,num_classes=3), 
                     batch_size=16, nb_epoch=75,validation_data=([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66]], np_utils.to_categorical(gt_test,num_classes=3)),verbose=2)

            pred = model.predict([test[:,:,0:12], test[:,:,12:24], test[:,:,24:26], test[:,:,26:40], test[:,:,40:54], test[:,:,54:66]], batch_size=32, verbose=2, steps=None)
            class_pred = pred.argmax(axis=-1)
            cnf_matrix = get_cnf_mat(gt_test,class_pred)

    return history, pred, cnf_matrix


