import numpy as np
import scipy as sp
import os
import scipy.ndimage, scipy.misc
""" Crop and pad depth images to 1920x1080 """

# output folder
outpath = "C:\\Users\\roh\\Desktop\\test\\test\\" 

# input folder
path = "C:\\Users\\roh\\Desktop\\test\\" 

imagelist = os.listdir(path)

for i in range(0, np.size(imagelist,0)-1):
    frame = scipy.ndimage.imread(path + imagelist[i]) # load image

    frame = scipy.misc.imresize(frame, 3) # resize image by 3.0 factor
    frame = np.delete(frame, [frame[0:95,:], frame[1080:frame.shape[0]-1,:]], axis=0) # crop top and bottom rows
    pad = np.zeros(1080,192)
    frame = np.concatenate((pad,frame,pad),axis=1) # pad left and right
    scipy.misc.imsave(outpath + imagelist[i],frame) # save image to output folder 
