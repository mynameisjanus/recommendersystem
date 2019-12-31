import scipy.io as sio
import numpy as np
import funcs
import EM_algorithm

# First we need to load the data which is in matlab format
# This returns a dictionary
ratings = sio.loadmat('ratings.mat')

# Extract the array of ratings
X = ratings['X']

log = np.empty([4, 5])
for i in range(4):
        for j in range(5):
                K = i + 1
                mixture, post = funcs.init(X, K, seed = j)
                mix, post, loglike = EM_algorithm.run(X, mixture, post)
                log[i,j] = loglike
        funcs.plot(X, mix, post, "Gaussian Mixture Model with K=" + str(K))
