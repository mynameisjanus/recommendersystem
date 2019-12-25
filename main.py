import scipy.io as sio
import numpy as np

# First we need to load the data which is in matlab format
ratings = sio.loadmat('ratings.mat')

# Convert to numpy array
X = ratings['X']

log = np.empty([4, 5])
for i in range(4):
        for j in range(5):
                K = i + 1
                mixture, post = common.init(X, K, seed = j)
                mix, post, loglike = naive_em.run(X, mixture, post)
                log[i,j] = loglike
        funcs.plot(X, mix, post, "GMM with K=" + str(K)).savefig("GMM" + str(K) + ".png")
