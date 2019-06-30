/** @file GammaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief GammaDist class definition
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
const GammaDist::NparamsVecT GammaDist::_param_lbound = {0, 0}; //Lower bound on valid parameter values 
const GammaDist::NparamsVecT GammaDist::_param_ubound = {INFINITY, INFINITY}; //Upper bound on valid parameter values


/* Constructors */
GammaDist::GammaDist(double scale, double shape) 
    : UnivariateDist(),
      _scale(checked_scale(scale)),
      _shape(checked_shape(shape)),
      llh_const_initialized(false)
{ }

/* Non-static member functions */

void GammaDist::set_scale(double val) 
{ 
    _scale = checked_scale(val); 
    llh_const_initialized = false;
}

void GammaDist::set_shape(double val) 
{ 
    _shape = checked_shape(val); 
    llh_const_initialized = false;
}

void GammaDist::set_params(double scale, double shape) 
{ 
    _scale = checked_scale(scale);  
    _shape = checked_shape(shape); 
    llh_const_initialized = false;
}

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

double GammaDist::llh(double x) const 
{ 
    if(!llh_const_initialized) initialize_llh_const();
    return rllh(x) + llh_const; 
}

void GammaDist::initialize_llh_const() const
{
    llh_const = compute_llh_const(shape(),scale());
    llh_const_initialized = true;
}

double GammaDist::compute_llh_const(double shape, double scale)
{
    return -shape*log(scale) - std::lgamma(shape);
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
