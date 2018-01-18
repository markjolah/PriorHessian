# PriorHessian
Library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined with Archemedian copulas.

## Installation

```
./build.sh
```

### Dependencies

* Armadillo
* Boost

### ExternalProject Dependencies

* BacktraceException - For debugging in Matlab plugins.

## Motivation

For many likelihood-based methods, they can be extended to Bayesian methods like MAP Estimation and MCMC Posterior sampling,
by incorporating a prior.  This prior must provide fast methods for computing log-likelihood and it's derivatives over the
parameter space.  The prior log-likelihood, as well as it's gradient and hessian are then added to the equivalent quantities from the likelihood
to create a Bayesian objective for MAP Estimation.

## Static Polymorphism

The PriorHessian library is designed using static polymorphism (templates), and as such avoids virtual functions for small-grained 
tasks, and instead uses templates, which allow many small functions to be inlined.  This aggressive inlining by the compiler
produces llh, grad, and hess functions that are nearly as fast as hand-coded functions.  But our flexible CompositeDist class
is able to be easily created with any mix of UnivariateDist and MultivariateDist elements.

In Mappel we use this ability to create heterogeneous priors for each Model's parameters, (e.g., [x,y,I,bg,sigma]).

Functionally, the PriorHessian library stores sequences of distributions as `std::tuples`.  Using this approach as opposed to
the runtime polymorpism of using `std::vector<std::unique_ptr<Base>>` gains several advantages.  
Most importantly, without the need for virtual functions, the tuple-based approach has the ability to inline the many
small computational functions that must be combined for every call to compute the log-likelihood or other computationally important quantities.

## Computations availible

 * `cdf` - cumulative distribution function
 * `pdf` - probability density function
 * `llh` - log-likelihood (log of pdf)
 * `rllh` - relative log-likelihood (log of pdf without constant terms)
 * `grad` - derivative of log-likelihood (or equivalently of relative-llh)
 * `grad2` - 2nd-derivative of log-likelihood

## Including PriorHessian as an ExternalProject

