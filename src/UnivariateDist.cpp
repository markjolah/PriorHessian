/** @file UnivariateDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief UnivariateDist base class method definition.
 * 
 * 
 */

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <cmath>
#include <sstream>

namespace prior_hessian {
    
void UnivariateDist::check_bounds(double lbound, double ubound)
{
    if( !(lbound < ubound) ){ //This comparison checks for NaNs    
        std::ostringstream msg;
        msg<<"UnivariateDist::set_bounds: Invalid bounds lbound:"<<lbound<<" ubound:"<<ubound;
        throw ParameterValueError(msg.str());
    }
}

} /* namespace prior_hessian */
