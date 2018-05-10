import numpy as np
import scipy as sp
import os

""" Crop and pad depth images to 1920x1080 """

# output folder
outpath = "" 

# input folder
path = "" 

imagelist = os.listdir(path)

for i in range(0, imagelist.shape[0]-1):
    frame = sp.ndimage.imread(path + imagelist[i]) # load image

    frame = sp.misc.imresize(frame, 3) # resize image by 3.0 factor
    frame = np.delete(frame, [frame[0:95,:], frame[1080:frame.shape[0]-1,:]], axis=0) # crop top and bottom rows
    pad = np.zeros(1080,192)
    frame = np.concatenate((pad,frame,pad),axis=1) # pad left and right
    sp.misc.imsave(outpath + imagelist[i],frame) # save image to output folder 
