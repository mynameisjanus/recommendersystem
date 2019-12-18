import scipy.io as sio

# First we need to load the data which is in matlab format
ratings = sio.loadmat('ratings.mat')

# Convert to numpy array
X = ratings['X']