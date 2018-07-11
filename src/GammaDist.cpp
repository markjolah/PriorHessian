/** @file GammaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief GammaDist class defintion
 * 
 */
#include "PriorHessian/GammaDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>
#include <limits>

#include <boost/math/special_functions/gamma.hpp>



namespace prior_hessian {

const StringVecT GammaDist::param_names = { "scale", "shape" };

/* Constructors */
GammaDist::GammaDist(double scale, double shape) 
    : UnivariateDist(0,INFINITY),
      _scale(check_scale(scale)),
      _shape(check_shape(shape)),
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
    return - _shape*log(_scale) - lgamma(_shape);
}

double GammaDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _scale;
        case 1:
            return _shape;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

void GammaDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_scale(val);
            return;
        case 1:
            set_shape(val);
            return;
        default:
            //Don't handle indexing errors.
            return;
    }
}

double GammaDist::check_scale(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"GammaDist: got bad scale value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

double GammaDist::check_shape(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"GammaDist: got bad shape value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}

} /* namespace prior_hessian */
