import json
import os
import numpy as np
import scipy as sp
import pandas as pd
from feature_smooth import feature_smooth

#Load 
path = "E:\\MATLAB\\Project\\Project\\keypoints_PAN\\"
json_files = os.listdir(path)
face_feats_all = np.zeros([2, len(json_files), 210], dtype=np.float64)
pose_feats_all = np.zeros([2, len(json_files), 54], dtype=np.float64)
pose_feats = np.zeros([len(json_files), 66], dtype=np.float64)

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
        face_feats_all[0,idx] = np.zeros([210])
        face_feats_all[1,idx] = np.zeros([210])
        pose_feats_all[0,idx] = np.zeros([54])
        pose_feats_all[1,idx] = np.zeros([54])
    
    # Similarity check for false positive detections;
    # check which candidate yields more keypoints, use the one that has
    # more
    k = np.count_nonzero([pose_feats_all[0,idx,0:2], pose_feats_all[0,idx,3:5], pose_feats_all[0,idx,42:44], pose_feats_all[0,idx,45:47], pose_feats_all[0,idx,6:8], pose_feats_all[0,idx,15:17]])
    a = np.count_nonzero([pose_feats_all[1,idx,0:2], pose_feats_all[1,idx,3:5], pose_feats_all[1,idx,42:44], pose_feats_all[1,idx,45:47], pose_feats_all[1,idx,6:8], pose_feats_all[1,idx,15:17]])

    if k < a:
        pose_feats_all[0,idx,:] = pose_feats_all[1,idx,:]
        face_feats_all[0,idx,:] = face_feats_all[1,idx,:]
    else:
        pass

    # Nose - Neck
    pose_feats[idx,0:2] = np.array([pose_feats_all[0,idx,0:2]])
    pose_feats[idx,2:4] = np.array([pose_feats_all[0,idx,3:5]])

    # REye - LEye
    pose_feats[idx,4:6] = np.array([pose_feats_all[0,idx,42:44]])
    pose_feats[idx,6:8] = np.array([pose_feats_all[0,idx,45:47]])

    # RShoulder - LShoulder
    pose_feats[idx,8:10] = np.array([pose_feats_all[0,idx,6:8]])
    pose_feats[idx,10:12] = np.array([pose_feats_all[0,idx,15:17]])

    # REye_refined
    pose_feats[idx,26:40] = np.ndarray.flatten(np.array([face_feats_all[0,idx,204:206], face_feats_all[0,idx,108:110], face_feats_all[0,idx,111:113],
                                       face_feats_all[0,idx,114:116], face_feats_all[0,idx,117:119], face_feats_all[0,idx,120:122], 
                                       face_feats_all[0,idx,123:125]]))

    # LEye_refined
    pose_feats[idx,40:54] = np.ndarray.flatten(np.array([face_feats_all[0,idx,207:209], face_feats_all[0,idx,126:128], face_feats_all[0,idx,129:131],
                                       face_feats_all[0,idx,132:134], face_feats_all[0,idx,135:137], face_feats_all[0,idx,138:140], 
                                       face_feats_all[0,idx,141:143]]))

    # facial keypoints if nose, REye or LEye is missing
    if not np.any(pose_feats[idx][0:2]):
        pose_feats[idx,0:2] = face_feats_all[0,idx,90:92]

    if not np.any(pose_feats[idx][4:5]):
        pose_feats[idx,4:6] = face_feats_all[0,idx,204:206]

    if not np.any(pose_feats[idx][6:7]):
        pose_feats[idx,6:8] = face_feats_all[0,idx,207:209]

# Interpolate for zero feature space elements (name is a bit misleading...)
pose_feats_smooth = feature_smooth(pose_feats)

# Calculate the rest of the feature space (distances, angles)
for i in range(0, len(pose_feats_smooth)):

    # Recalculate coordinates to nose origin
    pose_feats_smooth[i,2:4] = pose_feats_smooth[i,2:4] - pose_feats_smooth[i,0:2]
    pose_feats_smooth[i,4:6] = pose_feats_smooth[i,4:6] - pose_feats_smooth[i,0:2]
    pose_feats_smooth[i,6:8] = pose_feats_smooth[i,6:8] - pose_feats_smooth[i,0:2]
    pose_feats_smooth[i,8:10] = pose_feats_smooth[i,8:10] - pose_feats_smooth[i,0:2]
    pose_feats_smooth[i,10:12] = pose_feats_smooth[i,10:12] - pose_feats_smooth[i,0:2]
    pose_feats_smooth[i,26:40] = np.subtract(pose_feats_smooth[i,26:40].reshape((7,2)), pose_feats_smooth[i,0:2]).reshape((1,14))
    pose_feats_smooth[i,40:54] = np.subtract(pose_feats_smooth[i,40:54].reshape((7,2)), pose_feats_smooth[i,0:2]).reshape((1,14))

    pose_feats_smooth[i,0:2] = [0, 0]

    # Euclidean distance between all face features.
    pose_feats_smooth[i,12] = np.linalg.norm(pose_feats_smooth[i,0:2] - pose_feats_smooth[i,4:6])
    pose_feats_smooth[i,13] = np.linalg.norm(pose_feats_smooth[i,0:2] - pose_feats_smooth[i,6:8])
    pose_feats_smooth[i,14] = np.linalg.norm(pose_feats_smooth[i,4:6] - pose_feats_smooth[i,6:8])

    # Euclidean distance between neck and all face features.
    pose_feats_smooth[i,15] = np.linalg.norm(pose_feats_smooth[i,2:4] - pose_feats_smooth[i,0:2])
    pose_feats_smooth[i,16] = np.linalg.norm(pose_feats_smooth[i,2:4] - pose_feats_smooth[i,4:6])
    pose_feats_smooth[i,17] = np.linalg.norm(pose_feats_smooth[i,2:4] - pose_feats_smooth[i,6:8])

    # Euclidean distance between RShoulder and all face features.
    pose_feats_smooth[i,18] = np.linalg.norm(pose_feats_smooth[i,8:10] - pose_feats_smooth[i,0:2])
    pose_feats_smooth[i,19] = np.linalg.norm(pose_feats_smooth[i,8:10] - pose_feats_smooth[i,4:6])
    pose_feats_smooth[i,20] = np.linalg.norm(pose_feats_smooth[i,8:10] - pose_feats_smooth[i,6:8])

    # Euclidean distance between LShoulder and all face features.
    pose_feats_smooth[i,21] = np.linalg.norm(pose_feats_smooth[i,10:12] - pose_feats_smooth[i,0:2])
    pose_feats_smooth[i,22] = np.linalg.norm(pose_feats_smooth[i,10:12] - pose_feats_smooth[i,4:6])
    pose_feats_smooth[i,23] = np.linalg.norm(pose_feats_smooth[i,10:12] - pose_feats_smooth[i,6:8])

    # Angle between vec(neck,nose) and vec(neck,LShoulder)
    #u = np.cross([(pose_feats[i,2] - pose_feats[i,0]), ((pose_feats[i,3] - pose_feats[i,1])),0],[(pose_feats[i,2] - pose_feats[i,10]), ((pose_feats[i,3] - pose_feats[i,11])),0])

    #v = np.dot([(pose_feats[i,2] - pose_feats[i,0]), ((pose_feats[i,3] - pose_feats[i,1])),0],[(pose_feats[i,2] - pose_feats[i,10]), ((pose_feats[i,3] - pose_feats[i,11])),0])

    pose_feats_smooth[i,24] = np.arccos(np.clip(np.dot(np.linalg.norm(pose_feats_smooth[i,0:2] - pose_feats_smooth[i,2:4]),np.linalg.norm(pose_feats_smooth[i,8:10] - pose_feats_smooth[i,2:4])),-1.0,1.0))

    #pose_feats_smooth[i,25] = 