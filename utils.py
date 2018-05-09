import numpy as np

def randomize(pose_feats_final, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(pose_feats_final.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    pose_feats_final = pose_feats_final[permutation]
    labels = labels[permutation]
    return pose_feats_final, labels

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
    pose_feats_final, labels = randomize(pose_feats_final, labels)

    train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3), 66], dtype=np.float64)
    gt_train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3)])
    test = np.zeros([4, int(np.floor(len(pose_feats_final)/4)), 66], dtype=np.float64)
    gt_test = np.zeros([4, int(np.floor(len(pose_feats_final)/4))-1])
    #gt_train_mlp = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3),3])

    """ create subsets for training """
    train[0][:,:] = np.array(pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)*3),:])
    gt_train[0,:] = np.transpose(np.array(labels[0:int(np.floor(len(pose_feats_final)/4)*3)]))

    train[1][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1,:])
    gt_train[1,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1]))

    train[2][:,:] = np.concatenate([pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)),:], pose_feats_final[int(np.floor(len(pose_feats_final)/4)*2):len(pose_feats_final)-1,:]],0)
    gt_train[2,:] = np.transpose(np.concatenate([labels[0:int(np.floor(len(pose_feats_final)/4))], labels[int(np.floor(len(pose_feats_final)/4)*2):len(pose_feats_final)-1]]))

    train[3][:,:] = np.concatenate([pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)*2),:], pose_feats_final[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:]])
    gt_train[3,:] = np.transpose(np.concatenate([labels[0:int(np.floor(len(pose_feats_final)/4)*2)], labels[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1]]))


    """ create subsets for testing """
    test[0][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:])
    gt_test[0,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-2]))

    test[1][:,:] = np.array(pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)),:])
    gt_test[1,:] = np.transpose(np.array(labels[0:int(np.floor(len(pose_feats_final)/4))-1]))

    test[2][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)):int(np.floor(len(pose_feats_final)/4)*2),:])
    gt_test[2,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)):int(np.floor(len(pose_feats_final)/4)*2)-1]))

    test[3][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)*2):int(np.floor(len(pose_feats_final)/4)*3),:])
    gt_test[3,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)*2):int(np.floor(len(pose_feats_final)/4)*3)-1]))

    return test, train, gt_test, gt_train

def norm_feats(pose_feats_smooth):
    """ Normalize all features, leave out all [0,0] nose coordinates. """
    
    trainsub = pose_feats_smooth[:,2:66]

    for i in range(0 , np.size(trainsub, 1)):
        minc = np.min(trainsub[:,i])
        maxc = np.max(trainsub[:,i])
        trainsub[:,i] = (trainsub[:,i] - minc) / (maxc - minc)

    pose_feats_smooth[:,2:66] = trainsub

    return pose_feats_smooth

def sample(pose_feats, labels):

    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    idx2 = np.flatnonzero(labels == 2)

    dom = np.min([len(idx0), len(idx1), len(idx2)])

    n_idx0 = idx0[0:dom-2]
    n_idx1 = idx1[0:dom-2]
    n_idx2 = idx2[0:dom-2]

    n_pose_feats0 = pose_feats[n_idx0]
    n_pose_feats1 = pose_feats[n_idx1]
    n_pose_feats2 = pose_feats[n_idx2]

    pose_feats = np.concatenate([n_pose_feats0, n_pose_feats1, n_pose_feats2])
    labels = np.concatenate([labels[n_idx0], labels[n_idx1], labels[n_idx2]])

    return pose_feats, labels