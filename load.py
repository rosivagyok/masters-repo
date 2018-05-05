import json
import os
import numpy as np, h5py
import scipy.io as sp
import pandas as pd
from feature_smooth import feature_smooth
from utils import angle_between, cross_validation

def load():
    pose_feats_smooth = np.load('keypoints.npy')
    data = pd.read_excel('PANDORA_ATTENTION_LABELS.xlsx')
    #data = np.load('keypoints.npy')
    labels = np.array(data)
    labels = labels[:,1]
    test, train, gt_test, gt_train = cross_validation( pose_feats_smooth, labels)
    return test, train, gt_test, gt_train