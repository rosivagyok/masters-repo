import tensorflow as tf
import numpy as np
import scipy as sp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
from model_load import late_DNN, early_DNN, late_LSTM, early_LSTM, evaluate, evaluate_lstm, evaluate_flexible, simple_LSTM, late_DNN2, early_DNN2,late_DNN4, early_DNN4, late_DNN6, early_DNN6
from load import load
from utils import sample_lstm, reshape_seqlist

train_label = ''
simple = True
depth_label = True
test, train, gt_test, gt_train, depth_train, depth_test = load(train_label)
shape = 1
if train_label=='lstm':
    range0, range1, range2 = sample_lstm( gt_train[1,:], shape )
    trange0, trange1, trange2 = sample_lstm( gt_test[1,:], shape )

    train, depth_train, gt_train, test, depth_test, gt_test = reshape_seqlist(range0,range1,range2,
                                                                              trange0,trange1,trange2,
                                                                              train,depth_train,test,
                                                                              depth_test,shape)
else:   
    X_train = train[1][:,:]
    X_depth_train = depth_train[1][:,:]
    Y_train = gt_train[1,:]
    X_test = test[1][0:test[1].shape[0]-1,:]
    X_depth_test = depth_test[1][0:test[1].shape[0]-1,:]
    Y_test = gt_test[1]
"""X_train = np.reshape(train[1][:,:],[train[1][:,:].shape[0],shape,train[1][:,:].shape[1]])
X_depth_train = np.reshape(depth_train[1][:,:],[depth_train[1][:,:].shape[0],shape,depth_train[1][:,:].shape[1]])
Y_train = gt_train[1,:]
X_test = np.reshape(test[1][0:test[1].shape[0]-1,:],[test[1][0:test[1].shape[0]-1,:].shape[0],shape,test[1][0:test[1].shape[0]-1,:].shape[1]])
X_depth_test = np.reshape(depth_test[1][0:depth_test[1].shape[0]-1,:],[depth_test[1][0:depth_test[1].shape[0]-1,:].shape[0],shape,depth_test[1][0:depth_test[1].shape[0]-1,:].shape[1]])
Y_test = gt_test[1]"""

shape0 = 38
shape1 = 6
"""shape2 = 24
shape3 = 24
shape4 = 6
shape5 = 6"""

modelshape = 2

model = early_DNN2(shape0,shape1)

if train_label=='lstm':
    """history, pred, cnf_matrix = evaluate_lstm(model, train, gt_train, test, 
                            gt_test, depth_train, depth_test, depth_label, simple)"""

    history, pred, cnf_matrix = evaluate_lstm(model, X_train, Y_train, X_test, 
                            Y_test, X_depth_train, X_depth_test, depth_label,simple)
else:
    history, pred, cnf_matrix = evaluate_flexible(model, X_train, Y_train, X_test, 
                            Y_test, X_depth_train, X_depth_test, modelshape)