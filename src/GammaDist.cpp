/** @file GammaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief GammaDist class defintion
 * 
 */
#include "PriorHessian/GammaDist.h"
#include "PriorHessian/util.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>
#include <limits>

#include <boost/math/special_functions/gamma.hpp>



namespace prior_hessian {

const StringVecT GammaDist::_param_names = { "scale", "shape" };
const VecT GammaDist::_param_lbound = {0, 0}; //Lower bound on valid parameter values 
const VecT GammaDist::_param_ubound = {INFINITY, INFINITY}; //Upper bound on valid parameter values


/* Constructors */
GammaDist::GammaDist(double scale, double shape) 
    : UnivariateDist(0,INFINITY),
      _scale(checked_scale(scale)),
      _shape(checked_shape(shape)),
      llh_const(compute_llh_const())
{ }

/* Non-static member functions */
double GammaDist::cdf(double x) const
{
   return boost::math::gamma_p(_shape, x / _scale);
}

double GammaDist::icdf(double u) const
{
    if(u == 0) return 0;
    if(u == 1) return INFINITY;
    return boost::math::gamma_p_inv(_shape, u) * _scale;
}

double GammaDist::pdf(double x) const
{
    if(x==0) return 0;
    double inv_scale = 1/_scale;
    return boost::math::gamma_p_derivative(_shape, x*inv_scale) * inv_scale;
}

double GammaDist::compute_llh_const() const
{
    return - _shape*log(_scale) - std::lgamma(_shape);
}


double GammaDist::checked_scale(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"GammaDist: got bad scale value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

double GammaDist::checked_shape(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"GammaDist: got bad shape value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

} /* namespace prior_hessian */
