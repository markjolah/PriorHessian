/** @file UnivariateDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief UnivariateDist base class method definition.
 * 
 * 
 */

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>

namespace prior_hessian {
    
UnivariateDist::UnivariateDist(double lbound, double ubound)
    : _lbound(lbound),
      _ubound(ubound)
{ check_bounds(lbound,ubound); }

void UnivariateDist::check_bounds(double lbound, double ubound)
{
    if( !(lbound < ubound) ){ //This comparison checks for NaNs    
        std::ostringstream msg;
        msg<<"UnivariateDist::set_bounds: Invalid bounds lbound:"<<lbound<<" ubound:"<<ubound;
        throw ParameterValueError(msg.str());
    }
}

void UnivariateDist::set_bounds(double lbound, double ubound)
{
    if(lbound != _lbound || ubound != _ubound)
        throw InvalidOperationError("UnivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
}

void UnivariateDist::set_lbound(double lbound)
{
    if(lbound != _lbound)
        throw InvalidOperationError("UnivariateDist: Unable to set lbound.  This object is not scalable or truncatable.");
}

void UnivariateDist::set_ubound(double ubound)
{
    if(ubound != _ubound)
        throw InvalidOperationError("UnivariateDist: Unable to set ubound.  This object is not scalable or truncatable.");
}

} /* namespace prior_hessian */
