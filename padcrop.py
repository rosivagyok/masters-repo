import numpy as np
import scipy as sp
import os
import scipy.ndimage, scipy.misc
import cv2
""" Crop and pad depth images to 1920x1080 """

# output folder
outpath = "C:\\Users\\roh\\Desktop\\test\\test\\" 

# input folder
path = "C:\\Users\\roh\\Desktop\\test\\" 

imagelist = os.listdir(path)

for i in range(0, np.size(imagelist,0)-1):
    frame = scipy.ndimage.imread(path + imagelist[i]) # load image
    #frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

    frame = scipy.misc.imresize(frame, 300) # resize image by 3.0 factor
    frame = np.delete(frame, range(0,96), axis=0) # crop top and bottom rows , frame[1080:frame.shape[0]-1,:]], 
    frame = np.delete(frame, range(1080,frame.shape[0]), axis=0)
    pad = np.zeros([1080,192])
    frame = np.concatenate((pad,frame),axis=1) # pad left and right
    frame = np.concatenate((frame,pad),axis=1) # pad left and right
    scipy.misc.imsave(outpath + imagelist[i],frame) # save image to output folder 
