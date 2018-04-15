import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(u, v):
    """ Returns the angle in degrees between unit vectors 'u' and 'v'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(u)
    v2_u = unit_vector(v)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), 
                                        -1.0, 1.0)))

def cross_validation(pose_feats_smooth, labels):
    """ Returns normalized cross validation training and test sets. """

    """ Normalize all features """
    pose_feats_final = norm_feats(pose_feats_smooth)

    """ create subsets for training """
    train.train1 = pose_feats_final[0:np.floor(len(pose_feats_final)/4)*3,:]
    gt_train.gt_train1 = labels[0:np.floor(len(pose_feats_final)/4)*3]

    train.train2 = pose_feats_final[np.floor(len(pose_feats_final)/4):len(pose_feats_final)-1,:]
    gt_train.gt_train2 = labels[np.floor(len(pose_feats_final)/4):len(pose_feats_final)-1]

    train.train3 = np.array([pose_feats_final[0:np.floor(len(pose_feats_final)/4),:], pose_feats_final[np.floor(len(pose_feats_final)/4)*2:len(pose_feats_final)-1,:]])
    gt_train.gt_train3 = np.array([labels[0:np.floor(len(pose_feats_final)/4)], labels[np.floor(len(pose_feats_final)/4)*2:len(pose_feats_final)-1]])

    train.train4 = np.array([pose_feats_final[0:np.floor(len(pose_feats_final)/4)*2,:], pose_feats_final[np.floor(len(pose_feats_final)/4)*3:len(pose_feats_final)-1,:]])
    gt_train.gt_train4 = np.array([labels[0:np.floor(len(pose_feats_final)/4*3)], labels[np.floor(len(pose_feats_final)/4)*3:len(pose_feats_final)-1]])


    """ create subsets for testing """
    test.test1 = pose_feats_final[np.floor(len(pose_feats_final)/4)*3:len(pose_feats_final)-1,:]
    gt_test.gt_test1 = labels[np.floor(len(pose_feats_final)/4)*3:len(pose_feats_final)-1]

    test.test2 = pose_feats_final[0:np.floor(len(pose_feats_final)/4),:]
    gt_test.gt_test2 = labels[0:np.floor(len(pose_feats_final)/4):len(pose_feats_final)-1]

    test.test3 = pose_feats_final[np.floor(len(pose_feats_final)/4):np.floor(len(pose_feats_final)/4)*2,:]
    gt_test.gt_test3 = labels[np.floor(len(pose_feats_final)/4):np.floor(len(pose_feats_final)/4)*2]

    test.test4 = pose_feats_final[np.floor(len(pose_feats_final)/4)*2:np.floor(len(pose_feats_final)/4)*3,:]
    gt_test.gt_test4 = labels[np.floor(len(pose_feats_final)/4)*2:np.floor(len(pose_feats_final)/4)*3]

    """ mlp """
    gt_train.gt_train1_mlp = np.zeros(len(gt_train.gt_train1),3)
    gt_train.gt_train2_mlp = np.zeros(len(gt_train.gt_train2),3)
    gt_train.gt_train3_mlp = np.zeros(len(gt_train.gt_train3),3)
    gt_train.gt_train4_mlp = np.zeros(len(gt_train.gt_train4),3)
    
    gt_test.gt_test1_mlp = np.zeros(len(gt_test.gt_test1),3)
    gt_test.gt_test2_mlp = np.zeros(len(gt_test.gt_test2),3)
    gt_test.gt_test3_mlp = np.zeros(len(gt_test.gt_test3),3)
    gt_test.gt_test4_mlp = np.zeros(len(gt_test.gt_test4),3)

    idx0.sub1 = np.where(np.all(gt_train.gt_train1 == 0, axis=1))
    idx0.sub2 = np.where(np.all(gt_train.gt_train2 == 0, axis=1))
    idx0.sub3 = np.where(np.all(gt_train.gt_train3 == 0, axis=1))
    idx0.sub4 = np.where(np.all(gt_train.gt_train4 == 0, axis=1))
    idx1.sub1 = np.where(np.all(gt_train.gt_train1 == 1, axis=1))
    idx1.sub2 = np.where(np.all(gt_train.gt_train2 == 1, axis=1))
    idx1.sub3 = np.where(np.all(gt_train.gt_train3 == 1, axis=1))
    idx1.sub4 = np.where(np.all(gt_train.gt_train4 == 1, axis=1))
    idx2.sub1 = np.where(np.all(gt_train.gt_train1 == 2, axis=1))
    idx2.sub2 = np.where(np.all(gt_train.gt_train2 == 2, axis=1))
    idx2.sub3 = np.where(np.all(gt_train.gt_train3 == 2, axis=1))
    idx2.sub4 = np.where(np.all(gt_train.gt_train4 == 2, axis=1))

    gt_train.gt_train1_mlp[idx0.sub1,0] = 1
    gt_train.gt_train1_mlp[idx1.sub1,1] = 1
    gt_train.gt_train1_mlp[idx2.sub1,2] = 1
    
    gt_train.gt_train2_mlp[idx0.sub2,0] = 1
    gt_train.gt_train2_mlp[idx1.sub2,1] = 1
    gt_train.gt_train2_mlp[idx2.sub2,2] = 1
    
    gt_train.gt_train3_mlp[idx0.sub3,0] = 1
    gt_train.gt_train3_mlp[idx1.sub3,1] = 1
    gt_train.gt_train3_mlp[idx2.sub3,2] = 1
    
    gt_train.gt_train4_mlp[idx0.sub4,0] = 1
    gt_train.gt_train4_mlp[idx1.sub4,1] = 1
    gt_train.gt_train4_mlp[idx2.sub4,2] = 1

    return test, train, gt_test, gt_train

def norm_feats(pose_feats_smooth):
    """ Normalize all features, leave out all [0,0] nose coordinates. """
    
    trainsub = pose_feats_smooth[:,2:66]

    for i in np.size(trainsub, 1):
        minc = np.min(trainsub[:,i])
        maxc = np.max(trainsub[:,i])
        trainsub[:,i] = (trainsub[:,i] - minc) / (maxc - minc)

    pose_feats_smooth[:,2:66] = trainsub

    return pose_feats_smooth