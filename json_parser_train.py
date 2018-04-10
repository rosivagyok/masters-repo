import json
import os
import numpy as np
import scipy as sp
import pandas as pd
from feature_smooth import feature_smooth

#Load 
path = "E:\\MATLAB\\Project\\Project\\keypoints_PAN\\"
json_files = os.listdir(path)
face_feats_all = np.empty([2, len(json_files), 210], dtype=np.float64)
pose_feats_all = np.empty([2, len(json_files), 54], dtype=np.float64)
pose_feats = np.empty([len(json_files), 66], dtype=np.float64)

for idx in range(0,len(json_files)):
    data = json.load(open(path + json_files[idx]))

    if len(data['people']) > 0:
        
        face_feats_all[0,idx] = data['people'][0]['face_keypoints']
        pose_feats_all[0,idx] = data['people'][0]['pose_keypoints']
        try:
            face_feats_all[1,idx] = data['people'][1]['face_keypoints']
            pose_feats_all[1,idx] = data['people'][1]['pose_keypoints']
        except IndexError:
            pass

    else:
        face_feats_all[0,idx] = np.empty([210])
        face_feats_all[1,idx] = np.empty([210])
        pose_feats_all[0,idx] = np.empty([54])
        pose_feats_all[1,idx] = np.empty([54])
    
    # Similarity check for false positive detections;
    # check which candidate yields more keypoints, use the one that has
    # more
    k = np.count_nonzero([pose_feats_all[0,idx,0:2], pose_feats_all[0,idx,2:4], pose_feats_all[0,idx,42:44], pose_feats_all[0,idx,45:47], pose_feats_all[0,idx,6:8], pose_feats_all[0,idx,15:17]])
    a = np.count_nonzero([pose_feats_all[1,idx,0:2], pose_feats_all[1,idx,2:4], pose_feats_all[1,idx,42:44], pose_feats_all[1,idx,45:47], pose_feats_all[1,idx,6:8], pose_feats_all[1,idx,15:17]])

    if k < a:
        pose_feats_all[0,idx] = pose_feats_all[1,idx]
        face_feats_all[0,idx] = face_feats_all[1,idx]
    else:
        pass

    # Nose - Neck
    pose_feats[idx,0:2] = np.array([pose_feats_all[0,idx,0:2]])
    pose_feats[idx,2:4] = np.array([pose_feats_all[0,idx,2:4]])

    # REye - LEye
    pose_feats[idx,4:6] = np.array([pose_feats_all[0,idx,42:44]])
    pose_feats[idx,6:8] = np.array([pose_feats_all[0,idx,45:47]])

    # RShoulder - LShoulder
    pose_feats[idx,8:10] = np.array([pose_feats_all[0,idx,6:8]])
    pose_feats[idx,10:12] = np.array([pose_feats_all[0,idx,15:17]])

    # REye_refined
    pose_feats[idx][26:40] = np.ndarray.flatten(np.array([face_feats_all[0,idx,204:206], face_feats_all[0,idx,108:110], face_feats_all[0,idx,111:113],
                                       face_feats_all[0,idx,114:116], face_feats_all[0,idx,117:119], face_feats_all[0,idx,120:122], 
                                       face_feats_all[0,idx,123:125]]))

    # LEye_refined
    pose_feats[idx][40:54] = np.ndarray.flatten(np.array([face_feats_all[0,idx,207:209], face_feats_all[0,idx,126:128], face_feats_all[0,idx,129:131],
                                       face_feats_all[0,idx,132:134], face_feats_all[0,idx,135:137], face_feats_all[0,idx,138:140], 
                                       face_feats_all[0,idx,141:143]]))

    # facial keypoints if nose, REye or LEye is missing
    if not np.any(pose_feats[idx][0:2]):
        pose_feats[idx,0:2] = face_feats_all[0,idx,90:92]

    if not np.any(pose_feats[idx][4:5]):
        pose_feats[idx,4:6] = face_feats_all[0,idx,204:206]

    if not np.any(pose_feats[idx][6:7]):
        pose_feats[idx,6:8] = face_feats_all[0,idx,207:209]


pose_feats_smooth = feature_smooth(pose_feats)