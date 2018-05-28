import numpy as np
from sklearn.decomposition import PCA
from pca_trafo import pca_trafo

def randomize(pose_feats_final, d_list, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(pose_feats_final.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    pose_feats_final = pose_feats_final[permutation]
    labels = labels[permutation]
    d_list = d_list[permutation]
    return pose_feats_final, d_list, labels

def randomize_lstm(seq, d_seq, labels):
    # Generate the permutation index array.
    permutation = np.random.permutation(seq.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    seq = seq[permutation,:,:]
    labels = labels[permutation]
    d_seq = d_seq[permutation,:,:]
    return seq, d_seq, labels

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

def cross_validation(pose_feats_smooth, d_list, labels, train_label):
    """ Returns normalized cross validation training and test sets. """

    """ Normalize all features """
    pose_feats_final, d_list = norm_feats(pose_feats_smooth, d_list)
    if train_label=='':
        pose_feats_final, d_list, labels = randomize(pose_feats_final, d_list, labels)

    pca_trafo(pose_feats_final,comp=44)
    """pca_trafo(pose_feats_final,comp=8)
    pca_trafo(pose_feats_final,comp=6)
    pca_trafo(pose_feats_final,comp=4)
    pca_trafo(pose_feats_final,comp=2)"""
    model1 = PCA(n_components=44, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
    pose_feats_final = model1.fit_transform(pose_feats_final)

    """train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3), 66], dtype=np.float64)
    gt_train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3)])
    test = np.zeros([4, int(np.floor(len(pose_feats_final)/4)), 66], dtype=np.float64)
    gt_test = np.zeros([4, int(np.floor(len(pose_feats_final)/4))-1])
    depth_train = np.zeros([4, int(np.floor(len(d_list)/4)*3), 6], dtype=np.float64)
    depth_test = np.zeros([4, int(np.floor(len(d_list)/4)), 6], dtype=np.float64)"""

    train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3), 44], dtype=np.float64) #Y axes!!!!!
    gt_train = np.zeros([4, int(np.floor(len(pose_feats_final)/4)*3)])
    test = np.zeros([4, int(np.floor(len(pose_feats_final)/4)), 44], dtype=np.float64)
    gt_test = np.zeros([4, int(np.floor(len(pose_feats_final)/4))-1])
    depth_train = np.zeros([4, int(np.floor(len(d_list)/4)*3), 6], dtype=np.float64)
    depth_test = np.zeros([4, int(np.floor(len(d_list)/4)), 6], dtype=np.float64)

    """ create subsets for training """
    train[0][:,:] = np.array(pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)*3),:])
    depth_train[0][:,:] = np.array(d_list[0:int(np.floor(len(pose_feats_final)/4)*3),:])
    gt_train[0,:] = np.transpose(np.array(labels[0:int(np.floor(len(pose_feats_final)/4)*3)]))

    train[1][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1,:])
    depth_train[1][:,:] = np.array(d_list[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1,:])
    gt_train[1,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)):len(pose_feats_final)-1]))

    train[2][:,:] = np.concatenate([pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)),:], pose_feats_final[int(np.floor(len(pose_feats_final)/4)*2):len(pose_feats_final)-1,:]],0)
    depth_train[2][:,:] = np.concatenate([d_list[0:int(np.floor(len(pose_feats_final)/4)),:], d_list[int(np.floor(len(pose_feats_final)/4)*2):len(pose_feats_final)-1,:]],0)
    gt_train[2,:] = np.transpose(np.concatenate([labels[0:int(np.floor(len(pose_feats_final)/4))], labels[int(np.floor(len(pose_feats_final)/4)*2):len(pose_feats_final)-1]]))

    train[3][:,:] = np.concatenate([pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)*2),:], pose_feats_final[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:]])
    depth_train[3][:,:] = np.concatenate([d_list[0:int(np.floor(len(pose_feats_final)/4)*2),:], d_list[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:]])
    gt_train[3,:] = np.transpose(np.concatenate([labels[0:int(np.floor(len(pose_feats_final)/4)*2)], labels[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1]]))


    """ create subsets for testing """
    test[0][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:])
    depth_test[0][:,:] = np.array(d_list[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-1,:])
    gt_test[0,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)*3):len(pose_feats_final)-2]))

    test[1][:,:] = np.array(pose_feats_final[0:int(np.floor(len(pose_feats_final)/4)),:])
    depth_test[1][:,:] = np.array(d_list[0:int(np.floor(len(pose_feats_final)/4)),:])
    gt_test[1,:] = np.transpose(np.array(labels[0:int(np.floor(len(pose_feats_final)/4))-1]))

    test[2][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)):int(np.floor(len(pose_feats_final)/4)*2),:])
    depth_test[2][:,:] = np.array(d_list[int(np.floor(len(pose_feats_final)/4)):int(np.floor(len(pose_feats_final)/4)*2),:])
    gt_test[2,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)):int(np.floor(len(pose_feats_final)/4)*2)-1]))

    test[3][:,:] = np.array(pose_feats_final[int(np.floor(len(pose_feats_final)/4)*2):int(np.floor(len(pose_feats_final)/4)*3),:])
    depth_test[3][:,:] = np.array(d_list[int(np.floor(len(pose_feats_final)/4)*2):int(np.floor(len(pose_feats_final)/4)*3),:])
    gt_test[3,:] = np.transpose(np.array(labels[int(np.floor(len(pose_feats_final)/4)*2):int(np.floor(len(pose_feats_final)/4)*3)-1]))

    return test, train, gt_test, gt_train, depth_train, depth_test

def norm_feats(pose_feats_smooth, d_list):
    """ Normalize all features, leave out all [0,0] nose coordinates. """
    
    trainsub = pose_feats_smooth[:,2:66]

    for i in range(0 , np.size(trainsub, 1)):
        minc = np.min(trainsub[:,i])
        maxc = np.max(trainsub[:,i])
        trainsub[:,i] = (trainsub[:,i] - minc) / (maxc - minc)

    for j in range(0, d_list.shape[1]):
        minc = np.min(d_list[:,j])
        maxc = np.max(d_list[:,j])
        d_list[:,j] = (d_list[:,j] - minc) / (maxc - minc)

    pose_feats_smooth[:,2:66] = trainsub

    return pose_feats_smooth, d_list

def sample(pose_feats, d_list, labels):

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
    d_list = np.concatenate([d_list[n_idx0], d_list[n_idx1], d_list[n_idx2]])
    labels = np.concatenate([labels[n_idx0], labels[n_idx1], labels[n_idx2]])

    return pose_feats, d_list, labels

def sample_lstm( labels, shape ):
    idx0 = np.flatnonzero(labels == 0)
    idx1 = np.flatnonzero(labels == 1)
    idx2 = np.flatnonzero(labels == 2)

    dom = np.min([len(idx0), len(idx1), len(idx2)])

    n_idx0 = idx0[0:dom-1]
    n_idx1 = idx1[0:dom-1]
    n_idx2 = idx2[0:dom-1]

    """ Create 0 sequences """
    range0 = []
    myval1 = 0
    myval2 = 0
    for i in range(0, n_idx0.shape[0]):
        if i == n_idx0.shape[0] - shape - 1:
            break
        myval1 = idx0[i]
        myval2 = idx0[i + shape]
        if myval1 not in range0:
            if myval2 - myval1 == shape:
                range0 = np.append(range0, idx0[i:i + shape])



    """ Create 1 sequences """
    range1 = []
    myval1 = 0
    myval2 = 0
    for i in range(0, n_idx1.shape[0]):
        if i == n_idx1.shape[0] - shape - 1:
            break
        myval1 = idx1[i]
        myval2 = idx1[i + shape]
        if myval1 not in range1:
            if myval2 - myval1 == shape:
                range1 = np.append(range1, idx1[i:i + shape])     


    """ Create 2 sequences """
    range2 = []
    myval1 = 0
    myval2 = 0
    for i in range(0, n_idx2.shape[0]):
        if i == n_idx2.shape[0] - shape - 1:
            break
        myval1 = idx2[i]
        myval2 = idx2[i + shape]
        if myval1 not in range2:
            if myval2 - myval1 == shape:
                range2 = np.append(range2, idx2[i:i + shape])


    return range0, range1, range2

def reshape_seqlist(range0,range1,range2,
                    trange0,trange1,trange2,
                    train,depth_train,test,
                    depth_test,shape):

    train0 = train[1][range0.astype(int),:]
    train1 = train[1][range1.astype(int),:]
    train2 = train[1][range2.astype(int),:]

    train0 = np.reshape(train0,[np.int(train0.shape[0]/shape),shape,train0.shape[1]])
    train1 = np.reshape(train1,[np.int(train1.shape[0]/shape),shape,train1.shape[1]])
    train2 = np.reshape(train2,[np.int(train2.shape[0]/shape),shape,train2.shape[1]])

    dom = np.min([train0.shape[0],train1.shape[0],train2.shape[0]])
    train0 = train0[0:dom-1,:,:]
    train1 = train1[0:dom-1,:,:]
    train2 = train2[0:dom-1,:,:]

    full_train = np.concatenate((train0,train1,train2),axis=0)

    trdepth0 = depth_train[1][range0.astype(int),:]
    trdepth1 = depth_train[1][range1.astype(int),:]
    trdepth2 = depth_train[1][range2.astype(int),:]

    trdepth0 = np.reshape(trdepth0,[np.int(trdepth0.shape[0]/shape),shape,trdepth0.shape[1]])
    trdepth1 = np.reshape(trdepth1,[np.int(trdepth1.shape[0]/shape),shape,trdepth1.shape[1]])
    trdepth2 = np.reshape(trdepth2,[np.int(trdepth2.shape[0]/shape),shape,trdepth2.shape[1]])

    trdepth0 = trdepth0[0:dom-1,:,:]
    trdepth1 = trdepth1[0:dom-1,:,:]
    trdepth2 = trdepth2[0:dom-1,:,:]

    full_depth_train = np.concatenate((trdepth0,trdepth1,trdepth2),axis=0)

    gt_train0 = np.zeros([len(train0),1])
    gt_train1 = np.ones([len(train1),1])
    gt_train2 = np.reshape(np.repeat(2,len(train2)),(np.repeat(2,len(train2)).shape[0],1))

    full_gt_train = np.concatenate((gt_train0,gt_train1,gt_train2))

    full_train, full_depth_train, full_gt_train = randomize_lstm(full_train, full_depth_train, full_gt_train)

    test0 = test[1][trange0.astype(int),:]
    test1 = test[1][trange1.astype(int),:]
    test2 = test[1][trange2.astype(int),:]

    test0 = np.reshape(test0,[np.int(test0.shape[0]/shape),shape,test0.shape[1]])
    test1 = np.reshape(test1,[np.int(test1.shape[0]/shape),shape,test1.shape[1]])
    test2 = np.reshape(test2,[np.int(test2.shape[0]/shape),shape,test2.shape[1]])

    dom = np.min([test0.shape[0],test1.shape[0],test2.shape[0]])
    test0 = test0[0:dom-1,:,:]
    test1 = test1[0:dom-1,:,:]
    test2 = test2[0:dom-1,:,:]

    full_test = np.concatenate((test0,test1,test2),axis=0)

    tedepth0 = depth_test[1][trange0.astype(int),:]
    tedepth1 = depth_test[1][trange1.astype(int),:]
    tedepth2 = depth_test[1][trange2.astype(int),:]

    tedepth0 = np.reshape(tedepth0,[np.int(tedepth0.shape[0]/shape),shape,tedepth0.shape[1]])
    tedepth1 = np.reshape(tedepth1,[np.int(tedepth1.shape[0]/shape),shape,tedepth1.shape[1]])
    tedepth2 = np.reshape(tedepth2,[np.int(tedepth2.shape[0]/shape),shape,tedepth2.shape[1]])

    tedepth0 = tedepth0[0:dom-1,:,:]
    tedepth1 = tedepth1[0:dom-1,:,:]
    tedepth2 = tedepth2[0:dom-1,:,:]

    full_depth_test = np.concatenate((tedepth0,tedepth1,tedepth2),axis=0)

    gt_test0 = np.zeros([len(test0),1])
    gt_test1 = np.ones([len(test1),1])
    gt_test2 = np.reshape(np.repeat(2,len(test2)),(np.repeat(2,len(test2)).shape[0],1))

    full_gt_test = np.concatenate((gt_test0,gt_test1,gt_test2))

    full_test, full_depth_test, full_gt_test = randomize_lstm(full_test, full_depth_test, full_gt_test)

    return full_train, full_depth_train, full_gt_train, full_test, full_depth_test, full_gt_test