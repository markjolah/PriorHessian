/** @file ScaledSymmetricBetaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief SymmetricBetaDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_SCALEDSYMMETRICBETADIST_H
#define PRIOR_HESSIAN_SCALEDSYMMETRICBETADIST_H

#include "PriorHessian/SymmetricBetaDist.h"
#include "PriorHessian/ScaledDist.h"

namespace prior_hessian {

using ScaledSymmetricBetaDist = ScaledDist<SymmetricBetaDist>;

inline
ScaledSymmetricBetaDist make_scaled_symmetric_beta_dist(double beta, double lbound, double ubound)
{
    return {SymmetricBetaDist(beta),lbound,ubound};
}

namespace detail
{
    /* Type traits for a bounded and non-bounded versions of distribution expose properties 
     * of the class useful for SFINAE techniques. */
    template<>
    struct dist_adaptor_traits<SymmetricBetaDist> 
    {
        using bounds_adapted_dist = ScaledSymmetricBetaDist;
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    struct dist_adaptor_traits<ScaledSymmetricBetaDist> 
    {
        using bounds_adapted_dist = ScaledSymmetricBetaDist;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_SCALEDSYMMETRICBETADIST_H */
