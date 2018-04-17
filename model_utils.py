import random
import numpy as np
from glob import glob
from os.path import join

def load_batch(train, gt, batch_size):

    X_batch, Y_batch = [], []

    loaded = 0

    # generate a batch of random keypoints
    while loaded < batch_size:
        idx = np.random.randint(0, np.size(dataset_npy, 1))

        # features
        x = train[idx,:]

        # labels
        y = gt[idx]

        X_batch.append(x)
        Y_batch.append(y)

        loaded += 1

    return X_batch, Y_batch