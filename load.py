import json
import os
import numpy as np, h5py
import scipy.io as sp
import pandas as pd
from feature_smooth import feature_smooth
from utils import angle_between, cross_validation
from sklearn.decomposition import PCA

def load():
    pose_feats_smooth = np.load('keypoints.npy')
    #data = pd.read_excel('PANDORA_ATTENTION_LABELS.xlsx')
    data = np.load('labels.npy')
    labels = np.array(data)
    #labels = labels[:,1].astype(int)
    pca = PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)
    pca.fit(pose_feats_smooth)
    pose_feats = pca.transform(pose_feats_smooth)
    test, train, gt_test, gt_train = cross_validation( pose_feats, labels)
    return test, train, gt_test, gt_train