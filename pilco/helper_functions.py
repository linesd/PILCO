import numpy as np


def kl_divergence(mu, cov):
    """
    Computes the KL divergence between a multivariate normal with mean (mu) and
    covariance (cov) and a standard multivariate normal N(0,I)

    Parameters
    ----------
    mu: mean of the posterior {array}, shape (n_samples + 1)

    S: covariance of the posterior {matrix}, shape (n_samples + 1, n_samples + 1)

    Returns
    -------
    KL : value of KL divergence {float}

    """
    n = mu.shape[0]
    kl = 0.5 * (-np.log(np.linalg.det(cov)) - n + np.trace(cov) + np.dot(mu.T, mu))
    return kl