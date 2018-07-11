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

const StringVecT ParetoDist::param_names = { "alpha" };

/* Constructors */
ParetoDist::ParetoDist(double alpha, double lbound) 
    : UnivariateDist(check_lbound(lbound),INFINITY),
      _alpha(check_alpha(alpha)),
      llh_const(compute_llh_const())
{ }

/* Non-static member functions */
    
double ParetoDist::compute_llh_const() const
{
    return log(_alpha) + _alpha*log(lbound());
}

double ParetoDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _alpha;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

void ParetoDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_alpha(val);
            return;
        default:
            //Don't handle indexing errors.
            return;
    }
}

double ParetoDist::check_alpha(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"ParetoDist: got bad alpha value:"<<val<<" Should be positive";
        throw ParameterValueError(msg.str());
    }
    return val;
}

double ParetoDist::check_lbound(double val)
{
    if(!std::isfinite(val) || val <= 0) {
        std::ostringstream msg;
        msg<<"ParetoDist: got bad lbound value:"<<val<<" Should be positive";
        throw ParameterValueError(msg.str());
    }
    return val;
}
    
} /* namespace prior_hessian */
