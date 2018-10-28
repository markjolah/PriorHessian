/** @file ParetoDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief ParetoDist class defintion
 * 
 */
#include "PriorHessian/ParetoDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>
#include <limits>

namespace prior_hessian {

const StringVecT ParetoDist::_param_names = {"min", "alpha" };
const VecT ParetoDist::_param_lbound = {0, 0}; //Lower bound on valid parameter values 
const VecT ParetoDist::_param_ubound = {INFINITY, INFINITY}; //Upper bound on valid parameter values

/* Constructors */
ParetoDist::ParetoDist(double min, double alpha) 
    : UnivariateDist(checked_min(min),INFINITY),
      _alpha(checked_alpha(alpha)),
      llh_const(compute_llh_const())
{ }

ParetoDist::ParetoDist(const VecT &params) 
    : UnivariateDist(checked_min(params[0]),INFINITY),
      _alpha(checked_alpha(params[1])),
      llh_const(compute_llh_const())
{ }

/* Non-static member functions */
    
double ParetoDist::compute_llh_const() const
{
    return log(_alpha) + _alpha*log(lbound());
}

double ParetoDist::checked_alpha(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"ParetoDist: got bad alpha value:"<<val<<" Should be positive";
        throw ParameterValueError(msg.str());
    }
    return val;
}

double ParetoDist::checked_min(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"ParetoDist: got bad min value:"<<val<<" Should be positive";
        throw ParameterValueError(msg.str());
    }
    return val;
}

} /* namespace prior_hessian */
