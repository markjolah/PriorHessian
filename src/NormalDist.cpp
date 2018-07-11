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

/* Constructors */
NormalDist::NormalDist(double mu, double sigma) 
    : UnivariateDist(-INFINITY,INFINITY),
      _mu(mu),
      _sigma(sigma),
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

double NormalDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _mu;
        case 1:
            return _sigma;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

void NormalDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_mu(val);
            return;
        case 1:
            set_sigma(val);
            return;
        default:
            return; //Don't handle indexing errors.
    }
}

double NormalDist::check_mu(double val)
{
    if(!std::isfinite(val)) {
        std::ostringstream msg;
        msg<<"NormalDist: got bad mu value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

double NormalDist::check_sigma(double val)
{
    if(val<=0 || !std::isfinite(val)) {
        std::ostringstream msg;
        msg<<"NormalDist: got bad sigma value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}
    
} /* namespace prior_hessian */
