import tensorflow as tf
import numpy as np
import scipy as sp
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.utils import np_utils
from model_load import late_DNN, early_DNN, late_LSTM, early_LSTM, evaluate
from load import load
from utils import sample_lstm

test, train, gt_test, gt_train, depth_train, depth_test, labels = load()
shape = 8
range0, range1, range2 = sample_lstm( labels, shape )

train0 = np.zeros([range0.shape[1], shape, train[1][:,:].shape[1]])
train1 = np.zeros([range1.shape[1], shape, train[1][:,:].shape[1]])
train2 = np.zeros([range2.shape[1], shape, train[1][:,:].shape[1]])
for i in range(0, range0.shape[1]):
    train0[:, i, :]

"""X_train = train[1][:,:]
X_depth_train = depth_train[1][:,:]"""
X_train = np.reshape(X_train,(np.int(X_train.shape[0]/8), 8, X_train.shape[1]))
Y_train = gt_train[1,0::8]
Y_train = gt_train[1,:]
"""X_test = test[1][0:test[1].shape[0]-1,:]
X_depth_test = depth_test[1][0:test[1].shape[0]-1,:]"""
X_test = np.reshape(X_test[0:len(X_test)-7,:],(np.int(X_test.shape[0]/8), 8, X_test.shape[1]))
Y_test = gt_test[1,0:np.size(gt_test[1])-7:8]
Y_test = gt_test[1,:]

model = early_LSTM(shape,depth_label=False)

history, pred = evaluate(model, X_train, Y_train, X_test, 
                         Y_test, X_depth_train, X_depth_test)