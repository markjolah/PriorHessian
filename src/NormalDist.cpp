/** @file NormalDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief NormalDist class defintion
 * 
 */
#include "PriorHessian/NormalDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <sstream>
#include <cmath>
#include <limits>

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>

namespace prior_hessian {

/* Static member variables */
const double NormalDist::sqrt2 =  ::sqrt(2);
const double NormalDist::sqrt2pi_inv = 1./::sqrt(2*boost::math::double_constants::pi);
const double NormalDist::log2pi = ::log(2*boost::math::double_constants::pi);

const StringVecT NormalDist::param_names = { "mu", "sigma" };
const VecT NormalDist::param_lbound= {-INFINITY, -INFINITY}; //Lower bound on valid parameter values 
const VecT NormalDist::param_ubound= {INFINITY, INFINITY}; //Upper bound on valid parameter values



/* Constructors */
NormalDist::NormalDist(double mu, double sigma) 
    : UnivariateDist(-INFINITY,INFINITY),
      _mu(checked_mu(mu)),
      _sigma(checked_sigma(sigma)),
      sigma_inv(1/_sigma),
      llh_const(compute_llh_const())
{ }

/* Non-static member functions */
double NormalDist::cdf(double x) const
{
    return .5*(1+boost::math::erf((x - _mu)/(sqrt2 * _sigma)));
}

double NormalDist::icdf(double u) const
{
    return _mu+_sigma*sqrt2*boost::math::erf_inv(2*u-1);
}

double NormalDist::checked_mu(double val)
{
    if(!std::isfinite(val)) {
        std::ostringstream msg;
        msg<<"NormalDist: got bad mu value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

double NormalDist::checked_sigma(double val)
{
    if(val<=0 || !std::isfinite(val)) {
        std::ostringstream msg;
        msg<<"NormalDist: got bad sigma value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}
    
} /* namespace prior_hessian */
