/** @file TruncatedGammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief TruncatedGammaDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_TRUNCATEDGAMMADIST_H
#define PRIOR_HESSIAN_TRUNCATEDGAMMADIST_H

#include "PriorHessian/GammaDist.h"
#include "PriorHessian/TruncatedDist.h"

namespace prior_hessian {

using TruncatedGammaDist = TruncatedDist<GammaDist>;

inline
TruncatedGammaDist make_bounded_gamma_dist(double scale, double shape,  std::pair<double,double> bounds)
{
    return {GammaDist(scale, shape),bounds.first,bounds.second};
}

namespace detail
{
    /* Type traits for a bounded and non-bounded versions of distribution expose properties 
     * of the class useful for SFINAE techniques. */
    template<>
    struct dist_adaptor_traits<GammaDist> 
    {
        using bounds_adapted_dist = TruncatedGammaDist;
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    struct dist_adaptor_traits<TruncatedGammaDist> 
    {
        using bounds_adapted_dist = TruncatedGammaDist;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */
    
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TUNCATEDGAMMADIST_H */
