import numpy as np
import h5py

# First we need to load the data which is in matlab format
dat = h5py.File('ratings.mat', 'r')
ratings = dat.get('data/variable')

# Convert to numpy array
ratings = np.array(ratings)
