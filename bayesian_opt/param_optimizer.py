from typing import List, Tuple
import torch
import gpytorch as gp
from typing import Callable


class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=gp.kernels.RBFKernel()):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x):
        self.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = self.likelihood(self(x))
        return observed_pred.mean, observed_pred.variance

class HyperparamOptimizer:
    def __init__(
            self,
            opt_func: Callable,
            param_limits: List[Tuple[float, float]],
            X: torch.Tensor,
            y: torch.Tensor,
            lr: float=0.05,
            lengthscale: float=0.05,
            outputscale: float=1.0,
            noise: float=1e-3,
        ):

        self.opt_func = opt_func
        self.param_limits = param_limits
        self.num_params = len(param_limits)

        self.likelihood = gp.likelihoods.GaussianLikelihood()
        self.gp_model = ExactGPModel(X, y, self.likelihood)
        self.gp_model.covar_module.base_kernel.lengthscale = lengthscale
        self.gp_model.covar_module.outputscale = outputscale
        self.gp_model.likelihood.noise = noise

        self.optimizer = torch.optim.Adam([
            {'params': self.gp_model.covar_module.parameters()},
            {'params': self.gp_model.mean_module.parameters()},
            {'params': self.gp_model.likelihood.parameters()},
        ], lr=lr)

        self.mll = gp.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)


    def train(self, X, y, n_iter=100):
        self.gp_model.set_train_data(X, y, strict=False)
        self.gp_model.train()
        self.gp_model.likelihood.train()

        for i in range(n_iter):
            self.optimizer.zero_grad()
            output = self.gp_model(X)
            loss = -self.mll(output, y)
            loss.backward()
            self.optimizer.step()

    def lower_confidence_bound(self, x, kappa=2.0):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        mu, variance = self.gp_model.predict(x)
        sigma = variance.sqrt()
        return mu - kappa * sigma

    def next_x(self, x_candidates, kappa=2.0):
        lcb_values = self.lower_confidence_bound(x_candidates, kappa)
        next_index = torch.argmin(lcb_values)
        return x_candidates[next_index].view(1, -1)

    def get_best(self):
        space = self.get_candidates(num_points=1000)
        mu, var = self.gp_model.predict(space)
        best_index = torch.argmin(mu)
        return space[best_index], mu[best_index]

    def get_candidates(self, num_points=100):
        if self.num_params == 1:
            z = torch.linspace(
                self.param_limits[0][0],
                self.param_limits[0][1],
                steps=num_points
            ).unsqueeze(-1)
            return z
        z = torch.rand(num_points, self.num_params)
        for i, (low, high) in enumerate(self.param_limits):
            z[:, i] = z[:, i] * (high - low) + low
        return z

    def optimize(self, X, y, n_iter=10, train_iter=200, kappa=2.0, num_candidates=100):
        history = []
        for iteration in range(n_iter):
            self.train(X, y, n_iter=train_iter)

            x_candidates = self.get_candidates(num_points=num_candidates)
            means, vars = self.gp_model.predict(x_candidates)

            x_next = self.next_x(x_candidates, kappa)
            y_next = self.opt_func(x_next)

            X = torch.cat([X, x_next], dim=0)
            y = torch.cat([y, y_next.view(-1)], dim=0)

            curr_best_x, curr_best_y = self.get_best()


            history.append({
                'x_candidates': x_candidates,
                'means': means,
                'vars': vars,
                'X': X,
                'y': y,
                'x_next': x_next,
                'y_next': y_next,
                'best_x': curr_best_x,
                'best_y': curr_best_y,
            })




        return history, self.get_best(), X, y