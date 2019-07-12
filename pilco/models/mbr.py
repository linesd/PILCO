import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import NotFittedError
from pilco.models.rffKernel import RFFKernel
from pilco.helper_functions import kl_divergence

class BR():
    """
    Bayesian linear regression with radial basis functions where the kernel
    matrix is approximated with random features.

    Parameters
    ----------
    num_rff_samples : int
        Number of Monte Carlo samples per feature

    """

    def __init__(self, num_rff_samples=2000):
        self.num_rff_samples = num_rff_samples
        self.is_fitted = False

        # Instantiate RFF approximations to the design matrix
        self.kernel = RFFKernel(D=self.num_rff_samples)
        self.post_k_xs = RFFKernel(D=self.num_rff_samples)
        self.pre_k_xs = RFFKernel(D=self.num_rff_samples)

    def add_data(self, X, Y):
        """
        Adds new data to the regressor and fits the random features kernel.

        Parameters
        ----------
        X : Data {array, matrix}, shape (n_samples, n_features)

        Y : Data {array, matrix}, shape (n_samples, n_features)

        Returns
        -------
        None

        """
        self.X = X
        self.Y = Y
        data_dim = X.shape[1]
        self.kernel.fit(data_dim)
        self.post_k_xs.fit(data_dim)
        self.pre_k_xs.fit(data_dim)
        self.is_fitted = True

    def calculate_posterior(self, gamma=0.5, var_y=0.2, var_w=1):
        """
        Calculates the posterior over the weights given the data and the hyperparameters.

        Parameters
        ----------
        gamma : {float} random feature lengthscale

        var_y: {float} target variance

        var_w: {float} prior weight variance

        Returns
        -------
        mu: mean of the posterior {array}, shape (n_samples + 1)

        S: convariance of the posterior {matrix}, shape (n_samples + 1, n_samples + 1)

        phi: design matrix {matrix}, shape (n_samples, n_samples + 1)

        """
        self.var_y = var_y
        self.gamma = gamma

        if not self.is_fitted:
            raise NotFittedError('Must call BR.fit() with new data first.')

        phi = np.append(np.ones((self.X.shape[0], 1)), \
                        self.kernel.approx_kernel(self.X, self.X, gamma=self.gamma), axis=1)  # Append ones for bias
        pre_w = 1 / var_w * np.eye(len(self.X) + 1)  # prior covariance matrix
        # Mean and covariance matrix for weights given x and y
        self.S = np.linalg.inv((phi.T).dot(phi) / self.var_y + pre_w)  # posterior distribution covariance matrix
        self.mu = self.S.dot(phi.T).dot(self.Y) / self.var_y  # MAP weights to use in mean(y*)

        return self.mu, self.S, phi

    def optimise(self, maxiter=100, disp=False):
        """
        Optimises the variational upper bound using Nelder-Mead method.

        Parameters
        ----------
        maxiter : {int} maximum number of iterations of miminize

        disp: {bool} set to True to print convergence messages

        Returns
        -------
        gamma: {float} random feature lengthscale

        var_y: {float} target variance

        """

        if not self.is_fitted:
            raise NotFittedError('Must call BR.fit() before the kernel can be approximated.')

        opts = {'maxiter': maxiter, 'disp': disp}
        result = minimize(self._upper_bound, [0, 0], method="Nelder-Mead", options=opts)
        print(result.message)
        return np.exp(result.x)

    def _upper_bound(self, x0):
        """
        Variational upper bound (VUB): -E[log p(Y|f(X))] + KL(q(w)||p(w))
        where -E[log p(Y|f(X))] is the expectation of the Gaussian likelihood,
        and KL(q(w)||p(w)) is the KL divergence between posterior and prior over
        the weights.

        Parameters
        ----------
        x0 : {array}, shape (2,) (gamma, var_y)

        Returns
        -------
        VUB: {float} value of variational upper bound
        """

        gamma, var_y = np.exp(x0)
        mu, S, phi = self.calculate_posterior(gamma=gamma, var_y=var_y)
        kl = kl_divergence(mu, S)  # KL divergence between posterior and prior on the weights
        N = self.X.shape[0]
        log_p = -(N / 2) * np.log(2 * np.pi * var_y) - (1 / (2 * var_y)) * (
                    (self.Y - phi.dot(mu)) ** 2).mean()  # likelihood

        return -log_p + kl

    def sample_weights(self, num_samples=1):
        """
        Draws a sample of the weights from the posterior over the weights.

        Parameters
        ----------
        num_samples : {int} number of sets of weights to draw

        Returns
        -------
        weights : {array, matrix}, shape (n_samples + 1, num_samples)

        """

        return np.random.multivariate_normal(self.mu.squeeze(), self.S, num_samples)

    def post_one_step(self, xs, weights):
        """
        Predict for new values xs given a particular set of weights

        Parameters
        ----------
        xs : {float, array} x position(s) for prediction

        weights : {array} set of weights drawn from the posterior

        Returns
        -------
        ys : {float, array} predictions

        """

        phi_xs = np.append(np.ones((len(xs), 1)),
                           self.post_k_xs.approx_kernel(xs, self.X, self.gamma), axis=1)  # Append ones for bias
        ys = phi_xs.dot(weights.T)

        return ys

    def pred_one_step(self, xs):
        """
        Predict for new values xs from the predictive distribution.

        Parameters
        ----------
        xs : {float, array} x position(s) for prediction

        Returns
        -------
        mu : {float, array} MAP of predictive, shape (len(xs),)

        stdev : {float, array} standard deviation of the predictive, shape (len(xs),)

        """

        pred_phi_xs = np.append(np.ones((len(xs), 1)),
                                self.pre_k_xs.approx_kernel(xs, self.X, self.gamma), axis=1)  # Append ones for bias
        mu = pred_phi_xs.dot(self.mu)  # calculate mean(y*)
        stdev = (np.sum(pred_phi_xs.dot(self.S) * pred_phi_xs, axis=1) + self.var_y) ** 0.5  # calculate Var(y*)^0.5

        return mu, stdev