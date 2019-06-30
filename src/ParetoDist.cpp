/** @file ParetoDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief ParetoDist class definition
 * 
 */
#include "PriorHessian/ParetoDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>
#include <limits>

namespace prior_hessian {

const StringVecT ParetoDist::_param_names = {"min", "alpha" };
const ParetoDist::NparamsVecT ParetoDist::_param_lbound = {0, 0}; //Lower bound on valid parameter values 
const ParetoDist::NparamsVecT ParetoDist::_param_ubound = {INFINITY, INFINITY}; //Upper bound on valid parameter values

/* Constructors */
ParetoDist::ParetoDist(double min, double alpha) 
    : UnivariateDist(),
      _min(checked_min(min)),
      _alpha(checked_alpha(alpha)),
      llh_const_initialized(false)
{ }

/* Non-static member functions */
void ParetoDist::set_min(double val) 
{ 
    _min = checked_min(val);
    llh_const_initialized = false;    
}

void ParetoDist::set_alpha(double val) 
{ 
    _alpha = checked_alpha(val); 
    llh_const_initialized = false;
}

void ParetoDist::set_params(double min, double alpha) 
{ 
    _min = checked_min(min);
    _alpha = checked_alpha(alpha); 
    llh_const_initialized = false;
}

void ParetoDist::set_lbound(double lbound)
{ 
    _min = checked_min(lbound);
    llh_const_initialized = false;  //Pareto llh_const depends on lbound.
}

double ParetoDist::mean() const 
{ 
    return (_alpha <=1) ? INFINITY : _alpha*lbound()/(_alpha-1); 
}

double ParetoDist::median() const 
{ 
    return lbound() * std::pow(2,1/alpha()); 
}

double ParetoDist::llh(double x) const 
{ 
    if(!llh_const_initialized) initialize_llh_const();
    return rllh(x) + llh_const; 
}

void ParetoDist::initialize_llh_const() const
{
    llh_const = compute_llh_const(lbound(),alpha());
    llh_const_initialized = true;
}

double ParetoDist::compute_llh_const(double lbound, double alpha)
{
    return log(alpha) + alpha*log(lbound);
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
