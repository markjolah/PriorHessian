/** @file TruncatedParetoDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief ParetoDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_TRUNCATEDPARETODIST_H
#define PRIOR_HESSIAN_TRUNCATEDPARETODIST_H

#include "PriorHessian/ParetoDist.h"
#include "PriorHessian/UpperTruncatedDist.h"

namespace prior_hessian {

/* A bounded normal dist uses the TruncatedDist adaptor */
using TruncatedParetoDist = UpperTruncatedDist<ParetoDist>;

inline
TruncatedParetoDist make_bounded_pareto_dist(double alpha, double lbound, double ubound)
{
    return {ParetoDist(alpha,lbound),ubound};
}

namespace detail
{
    /* Type traits for a bounded and non-bounded versions of distribution expose properties 
     * of the class useful for SFINAE techniques. */
    template<>
    struct dist_adaptor_traits<ParetoDist> 
    {
        using bounds_adapted_dist = TruncatedParetoDist;
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    struct dist_adaptor_traits<TruncatedParetoDist> 
    {
        using bounds_adapted_dist = TruncatedParetoDist;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */
    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_TUNCATEDPARETODIST_H */
