/** @file NormalDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief NormalDist class definition
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
const StringVecT NormalDist::_param_names = { "mu", "sigma" };
const NormalDist::NparamsVecT NormalDist::_param_lbound = {-INFINITY, 0}; //Lower bound on valid parameter values 
const NormalDist::NparamsVecT NormalDist::_param_ubound = {INFINITY, INFINITY}; //Upper bound on valid parameter values



/* Constructors */
NormalDist::NormalDist(double mu, double sigma) 
    : UnivariateDist()
{ 
    set_params(mu,sigma);
}

/* Non-static member functions */

void NormalDist::set_sigma(double val) 
{ 
    _sigma = checked_sigma(val); 
    _sigma_inv = 1./_sigma;
    llh_const_initialized = false;
}

double NormalDist::cdf(double x) const
{
    return .5*(1 + boost::math::erf((x - _mu)*_sigma_inv*constants::sqrt2_inv));
}

double NormalDist::icdf(double u) const
{
    return mu() + sigma()*constants::sqrt2*boost::math::erf_inv(2*u-1);
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

double NormalDist::llh(double x) const 
{ 
    if(!llh_const_initialized) initialize_llh_const(); //Lazy computation of llh_const.
    return rllh(x) + llh_const;
}

void NormalDist::initialize_llh_const() const
{
    llh_const = compute_llh_const(sigma());
    llh_const_initialized = true;
}

double NormalDist::compute_llh_const(double sigma)
{
    return -log(sigma) - .5*constants::log2pi;
}

} /* namespace prior_hessian */
