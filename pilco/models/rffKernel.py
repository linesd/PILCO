import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


# Random Fourier features class
class RFFKernel(BaseEstimator):
    """
    Approximates the feature map of a kernel by Monte Carlo approximation of
    it's Fourier transform. Implements random Fourier features [1].

    Parameters
    ----------
    gamma : float
        Parameter of the Gaussian kernel exp(-gamma * w^2)

    D : int
        Number of Monte Carlo samples per feature

    [1] Rahimi, A., & Recht, B. (2008). Random features for large-scale kernel machines.
        In Advances in neural information processing systems (pp. 1177-1184)
    """

    def __init__(self, D=50):
        self.D = D
        self.is_fitted = False

    def fit(self, data_dim):
        """
        Draws D samples for direction w and random offset b

        X : Data {array, matrix}, shape (n_samples, n_dimension)

        Returns
        -------
        self : object
            Returns the direction vector w, the offset b and the boolean
            fitted.
        """

        self.w_direction = np.random.normal(size=(self.D, data_dim))  # *np.sqrt(2 * self.gamma)
        self.b_offset = np.random.uniform(0, 2 * np.pi, size=(1, self.D))
        self.is_fitted = True

        return self

    def _transform(self, X):
        """
        Apply the approximate feature map to X.

        Parameters
        ----------
        X : Data {array, matrix}, shape (n_samples, n_features)

        Returns
        -------
        Z : array of transformed features, shape (n_samples, n_components [D])
        """

        Xw = X.dot(self.w_direction.T)

        Z = np.sqrt(2 / self.D) * np.cos(Xw + self.b_offset)

        return Z

    def approx_kernel(self, X, Y=None, gamma=1):
        """
        Computes the kernel gram matrix using the transformed Fourier features

        Parameters
        ----------
        X : Data {array, matrix}, shape (n_samples, n_features)

        Y : Data {array, matrix}, shape (n_samples, n_features)

        gamma: lengthscale {float}

        Returns
        -------
        K : gram matrix (n_samples, n_samples)
        """
        if not self.is_fitted:
            raise NotFittedError('Must call .fit(X) before the kernel can be approximated.')

        Zx = self._transform(X)
        if Y is not None:
            Zy = self._transform(Y)
            K = Zx.dot(Zy.T) * np.sqrt(2 * gamma)
        else:
            K = Zx.dot(Zx.T) * np.sqrt(2 * gamma)

        return K