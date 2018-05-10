import numpy as np
import scipy as sp
import cv2 as cv

def get_depth(pose_feats_points, imagelist):
    """ Get corresponding depth values of neighbours at each keypoint,
        and calculate the mean. """

    depth_k = np.zeros(pose_feats_points.shape[0],1)
    kernel = np.array([[1, 1, 1]
                       [1, 1, 1]
                       [1, 1, 1]])
    for i in range(0, pose_feats_points.shape[0]-1):

        x, y = pose_feats_points[i]
        point = np.array([[x-1, y-1], [x, y-1], [x+1, y-1]], 
                         [[x-1, y], [x, y], [x+1, y]],
                         [[x-1, y+1], [x, y+1], [x+1, y+1]])

        """Read image and get color intensity values"""
        folder = "E:\\keypoints\\full\\"
        frame_n = folder + imagelist[i] + '.png'
        frame = cv.imread(frame_n)
        point_i = frame[point]

        depth_k[i] = np.mean(np.matmul(kernel, point_i))

    #Normalize depth values
    minc = np.min(depth_k)
    maxc = np.max(depth_k)
    depth_k[:,i] = (depth_k - minc) / (maxc - minc)

    return depth_k

def depthlist(pose_feats_smooth, imagelist):

    d_list = np.zeros(pose_feats_points.shape[0],6)
    i = 0
    for j in range(0, 12, 2):
        pose_feats_points = pose_feats_smooth[:,j:j+1]
        list = get_depth(pose_feats_points, imagelist)
        d_list[:,i] = list

    return d_list