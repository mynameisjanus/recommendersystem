import scipy.io as sio
import numpy as np
from typing import NamedTuple, Tuple
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc

# First we need to load the data which is in matlab format
ratings = sio.loadmat('ratings.mat')

# Convert to numpy array
X = ratings['X']

class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()
#        var[j] = np.maximum(0.25, var[j]) #added line

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      0,
                      theta,
                      theta + offset,
                      edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()

for i in range(1, 5):
        for j in range(5):
                mixture, post = common.init(X, i, j)
                mix, pos, COST = kmeans.run(X, mixture, post)
                cost[i-1][j] = COST

print("Cost: ", cost)

for k in range(1,5):
        m2, p = common.init(X, k, seed = 4)
        m2, p, c = kmeans.run(X, m2, p)
        common.plot(X, m2, p, "K-Means with K = " + str(k))

log = np.empty([4, 5])
for i in range(4):
        for j in range(5):
                K = i + 1
                mixture, post = common.init(X, K, seed = j)
                mix, post, loglike = naive_em.run(X, mixture, post)
                log[i,j] = loglike
        common.plot(X, mix, post, "GMM with K=" + str(K)).savefig("GMM" + str(K) + ".png")
