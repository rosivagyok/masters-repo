import numpy as np
from scipy.interpolate import interp1d
from itertools import chain

def feature_smooth(pose_feats):
    for j in chain(range(0, 10, 2), range(26, len(pose_feats), 2)):
        idx = np.flatnonzero(pose_feats[:,j] == 0)

        k = 0;
        myval1 = 0
        myval2 = 0
        range_enum = np.zeros([len(idx),2])
       
        for i in range(0, len(idx)):
            if i == len(idx)-1:
                mayval2 = idx[i]
                range_enum[k,0:2] = np.array([myval1, myval2])
            elif np.logical_and(idx[i+1] - idx[i] == 1, myval1 == 0): 
                myval1 = idx[i]
            elif idx[i+1] - idx[i] == 1:
                pass
            else:
                myval2 = idx[i]
                range_enum[k,0:2] = np.array([myval1, myval2]) 
                k+=1
                myval1 = 0
                myval2 = 0
            
            if np.logical_and(k > 0, range_enum[k-1,0] == 0):
                range_enum[k-1,0] = range_enum[k-1,1]
            else:
                pass
        
        # In case we missed an element and it's zero, correct it
        miss = np.flatnonzero(range_enum[:,0] == 0)
        range_enum[miss,0] = range_enum[miss,1]

        # Delete zero rows
        range_enum = range_enum[~(range_enum==0).all(1)]

        for l in range(0, len(range_enum)):
            pathXY = np.concatenate(([pose_feats[range_enum[l, 0].astype(int) - 1, j:j+2]], [pose_feats[range_enum[l, 1].astype(int) + 1, j:j+2]]),0)
            stepLengths = np.sqrt(np.sum(np.array([(pathXY[1,0] - pathXY[0,0]), (pathXY[1,1] - pathXY[0,1])])**2, 0))
            stepLengths = np.array([0,stepLengths])
            cumulativeLen = np.cumsum(stepLengths)
            finalStepLocs = np.linspace(0, cumulativeLen[-1], 100)
            finalPathXY = interp1d(cumulativeLen,pathXY,axis=0)(finalStepLocs)

            a = np.arange(range_enum[l,0],range_enum[l,1],1)
            b = len(finalPathXY)/len(a)-1;
            c = finalPathXY[np.arange(b,len(a)*b,b),0:2]

            pose_feats[a,j:j+1] = c

        pose_feats_smooth = pose_feats

    return pose_feats_smooth