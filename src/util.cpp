/** @file util.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief Utilities
 */

#include "PriorHessian/util.h"

namespace prior_hessian {
    
namespace constants {
    const double sqrt2 = std::sqrt(2.);
    const double sqrt2_inv = 1./std::sqrt(2.);
    const double sqrt2pi = std::sqrt(2.*arma::datum::pi);
    const double sqrt2pi_inv = 1./std::sqrt(2.*arma::datum::pi);
    const double log2pi = std::log(2.*arma::datum::pi);    
} /* namespace prior_hessian::constants */

} /* namespace prior_hessian */
