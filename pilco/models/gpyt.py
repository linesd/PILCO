import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
import torch
import math

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :], likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



def trainGPyTorch(model, likelihood, X, Y, training_iterations, verbose=False):
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Use the adam optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X)
        # Calc loss and backprop derivatives
        loss = -mll(output, Y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()

    print("GPyTorch optimisation complete ...")

def run_uncertainty(pilco, likelihood, X, Y, holdout=False):

    if holdout:
        ind = int(math.floor(0.8 * len(X[0])))
        # pytorch stuff from here on.
        Xt = torch.from_numpy(X[0:ind, :]).float()
        Yt = torch.from_numpy(Y[0:ind, :]).float()

        Xh = torch.from_numpy(X[ind:, :]).float()
        Yh = torch.from_numpy(Y[ind:, :]).float()

    else:
        # pytorch stuff from here on.
        Xt = torch.from_numpy(X).float()
        Yt = torch.from_numpy(Y).float()

    active_models = []
    for i in range(pilco.mgpr.num_outputs):
        active_models.append(GPRegressionModel(Xt, Yt[:, i:i + 1].squeeze(), likelihood))
        trainGPyTorch(active_models[i], likelihood, Xt, Yt[:, i:i + 1].squeeze(), 100)

    if holdout:
        for i in range(pilco.mgpr.num_outputs):
            active_models[i].eval()
            likelihood.eval()
            preds = active_models[i](Xh)
            print('Model %s Test MAE: %.5f' % (i, torch.mean(torch.abs(preds.mean - Yh[:, i:i + 1]))))