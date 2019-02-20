<a href="https://travis-ci.org/markjolah/PriorHessian"><img src="https://travis-ci.org/markjolah/PriorHessian.svg?branch=master"/></a>

# Prior Hessian
Library for fast computation of log-likelihoods, and derivatives, of multivariate priors defined as composites of univariate distributions and multivariate distributions especially
Archimedean copulas.

## Documentation
The PriorHessian Doxygen documentation can be build with the `OPT_DOC` CMake option and is also available on online:
  * [PriorHessian HTML Manual](https://markjolah.github.io/PriorHessian/index.html)
  * [PriorHessian PDF Manual](https://markjolah.github.io/PriorHessian/pdf/PriorHessian-0.2-reference.pdf)
  * [PriorHessian github repository](https://github.com/markjolah/PriorHessian)

## Installation

The PriorHessian library uses CMake and is designed to be installed either to a base system, or run from within an arbitrary install prefix.  The default build script will install to the `_install` directory underneath the repository root.

    $ ./build.sh <cmake-extra-opts>


## Dependencies

* [*Armadillo*](http://arma.sourceforge.net/docs.html) - A high-performance array library for C++.
* *BLAS* - A BLAS implemenation: [Netlib BLAS reference](http://www.netlib.org/blas/) or [*OpenBlas*](https://www.openblas.net/)
* *LAPACK* - A Lapack implemenation: [Netlib LAPACK reference](http://www.netlib.org/lapack/)

Note the `OPT_BLAS_INT64` CMake option controls whether Armadillo uses BLAS and LAPACK libraries that use 64-bit interger indexing.
Matlab uses 64-bit by default, and to link PriorHessian to Matlab MEX libraries, this option must be on.  Many linux systems only provide 32-bit integer versions of BLAS and Lapack, and the option can be disabled if Matlab support is not a concern and 64-bit support is difficult to manage on

### External Projects
These packages are specialized CMake projects.  If they are not currently installed on the development machines we use the [AddExternalDependency.cmake](https://github.com/markjolah/UncommonCMakeModules/blob/master/AddExternalDependency.cmake) which will automatically download, configure, build and install to the `CMAKE_INSTALL_PREFIX`, enabling their use through the normal CMake `find_package()` system.

* [BacktraceException](https://github.com/markjolah/BacktraceException) - For exception backtraces when debugging (especially in Matlab).

## Motivation

For many likelihood-based methods, they can be extended to Bayesian methods like MAP Estimation and MCMC Posterior sampling,
by incorporating a prior.  This prior must provide fast methods for computing log-likelihood and it's derivatives over the
parameter space.  The prior log-likelihood, as well as it's gradient and hessian are then added to the equivalent quantities from the likelihood
to create a Bayesian objective for MAP Estimation.

## Static Polymorphism

The PriorHessian library is designed using static polymorphism (templates), and as such avoids virtual functions for small-grained  tasks, and instead uses templates, which allow many small functions to be inlined.  This aggressive inlining by the compiler produces log-likelihood, gradient, and hessian functions that are nearly as fast as hand-coded functions.  But our flexible [`CompositeDist`]() class is able to be easily created with any mix of [`UnivariateDist`]() and [`MultivariateDist`]() elements.

In [Mappel](https://github.com/markjolah/Mappel) we use this ability to create heterogeneous priors for each Model's parameters, (e.g., [x,y,I,bg,sigma]).

Functionally, the PriorHessian library stores sequences of distributions as `std::tuples`.  Using this approach as opposed to
the runtime polymorphism of using `std::vector<std::unique_ptr<Base>>` gains several advantages.
Most importantly, without the need for virtual functions, the tuple-based approach has the ability to inline the many
small computational functions that must be combined for every call to compute the log-likelihood or other computationally important quantities.

## Computations available

 * `cdf` - cumulative distribution function
 * `pdf` - probability density function
 * `llh` - log-likelihood (log of pdf)
 * `rllh` - relative log-likelihood (log of pdf without constant terms)
 * `grad` - derivative of log-likelihood (or equivalently of relative-llh)
 * `grad2` - 2nd-derivative of log-likelihood

## Including PriorHessian as an ExternalProject

