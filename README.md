# Transformers-for-Bayesian-statistics

## TODO 23.10. Write the basic theory and equations:
- What is a stochastic process (elementary properties of a process)
- define GP as a special case - mean, covariance, kernel...
- mention what the marginal likelihood is and why it is useful for parameter estimation

## TODO 17.10: toy problem for GP

1. manual GP: learn how it works under the hood:
    - choose a kernel function K(x,x') e.g. Square exponential with selected parameters
    - perform a GP regression around point [x,f(x)]=[1,1]
    - draw mean and variance of the GP on a grid of values for x', ie.e. p(x'|x,f(x))
  
    - do a sensitivity study for the parameters. Show how the prediction changes for different parameter values


2. toolbox GP
    - choose a test function e.g. f(x) = cos(x)*exp(-0.1x) on grid [-5,5]
    - select a few points for the function and predict the rest using GP from a toolbox (GPyTorch)
    - estimate kernel hyperparameters using Type II maximum likelihood
  
    - show how the approximation evolves with increasing number of samples from the true function

