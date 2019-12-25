"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post1 = np.zeros((n, K))

    mu = mixture.mu
    var = mixture.var
    p = mixture.p
    for i in range(n):
            for j in range(K):
                    post1[i,j] = p[j]*(1/(np.sqrt((2*np.pi*var[j])**d)))*np.exp(-np.linalg.norm(X[i,:]-mu[j])**2/(2*var[j]))
    post = post1 / post1.sum(axis=1, keepdims=True)
    pointlog = np.log(np.sum(post1,axis=1))
    loglike = np.sum(pointlog)
    return post, loglike
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    n_hat = np.sum(post, axis=0)
    dnhat = np.divide(1,n_hat)
    p = n_hat / n
    mu = np.dot(post.T,X) * dnhat[:,None] # K x d array
    norm = np.zeros((n,K))
    for i in range(n):
            for j in range(K):
                    norm[i,j] = (post[i,j] * (np.linalg.norm(X[i]-mu[j]))**2) * dnhat[j] / d
    var = np.sum(norm, axis=0)
    return GaussianMixture(mu, var, p)
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    post, oldlog = estep(X, mixture)
    mixture = mstep(X, post)
    post, loglike = estep(X, mixture)
    while (loglike - oldlog > 1e-6 * abs(loglike)):
            oldlog = loglike
            mixture = mstep(X, post)
            post, loglike = estep(X, mixture)
    return mixture, post, loglike
    raise NotImplementedError
