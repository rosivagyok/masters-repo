import json
import os
import numpy as np, h5py
import scipy.io as sp
import pandas as pd
from feature_smooth import feature_smooth
from utils import angle_between, cross_validation, sample
from sklearn.decomposition import PCA

def load(train_label):
    pose_feats = np.load('keypoints.npy')
    d_list = np.load('d_list.npy')
    data = np.load('labels.npy')
    labels = np.array(data)
    labels = np.append(labels,[0])

    if train_label=='':
        pose_feats, d_list, labels = sample(pose_feats, d_list, labels)

    test, train, gt_test, gt_train, depth_train, depth_test = cross_validation( pose_feats, d_list, labels, train_label)
    return test, train, gt_test, gt_train, depth_train, depth_test