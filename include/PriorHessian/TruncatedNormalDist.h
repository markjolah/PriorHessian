/** @file TruncatedNormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief TruncatedNormalDist class declaration.
 * 
 */

#ifndef PRIOR_HESSIAN_TRUNCATEDNORMALDIST_H
#define PRIOR_HESSIAN_TRUNCATEDNORMALDIST_H

#include "PriorHessian/NormalDist.h"
#include "PriorHessian/TruncatedDist.h"

namespace prior_hessian {

/* A bounded normal dist uses the TruncatedDist adaptor */
using TruncatedNormalDist = TruncatedDist<NormalDist>;

inline
TruncatedNormalDist make_bounded_normal_dist(double mu, double sigma, std::pair<double,double> bounds)
{
    return {NormalDist(mu, sigma),bounds.first,bounds.second};
}

namespace detail
{
    /* Type traits for a bounded and non-bounded versions of distribution expose properties 
     * of the class useful for SFINAE techniques. */
    template<>
    struct dist_adaptor_traits<NormalDist> 
    {
        using bounds_adapted_dist = TruncatedNormalDist;
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    struct dist_adaptor_traits<TruncatedNormalDist> 
    {
        using bounds_adapted_dist = TruncatedNormalDist;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */
    
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TRUNCATEDNORMALDIST_H */
