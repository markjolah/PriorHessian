/** @file TruncatedMultivariateNormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief TruncatedMultivariateNormalDist class declaration.
 * 
 */

#ifndef PRIOR_HESSIAN_TRUNCATEDMULTIVARIATENORMALDIST_H
#define PRIOR_HESSIAN_TRUNCATEDMULTIVARIATENORMALDIST_H

#include "PriorHessian/MultivariateNormalDist.h"
#include "PriorHessian/TruncatedMultivariateDist.h"

namespace prior_hessian {

/* A bounded normal dist uses the TruncatedMultivariateDist adaptor */
template<IdxT Ndim>
using TruncatedMultivariateNormalDist = TruncatedMultivariateDist<MultivariateNormalDist<Ndim>>;

template<IdxT Ndim, class Vec, class Mat, class Vec2>
TruncatedMultivariateNormalDist<Ndim> 
make_bounded_multivariate_normal_dist(Vec &&mu, Mat &&sigma, Vec2 &&lbound, Vec2 &&ubound)
{
    return {MultivariateNormalDist<Ndim>{std::forward<Vec>(mu), std::forward<Mat>(sigma)}, 
                std::forward<Vec2>(lbound),std::forward<Vec2>(ubound)};
}

namespace detail
{
    /* Type traits for a bounded and non-bounded versions of distribution expose properties 
     * of the class useful for SFINAE techniques. */
    template<IdxT Ndim>
    struct dist_adaptor_traits<MultivariateNormalDist<Ndim>> 
    {
        using bounds_adapted_dist = TruncatedMultivariateNormalDist<Ndim>;
        static constexpr bool adaptable_bounds = false;
    };
    
    template<IdxT Ndim>
    struct dist_adaptor_traits<TruncatedMultivariateNormalDist<Ndim>> 
    {
        using bounds_adapted_dist = TruncatedMultivariateNormalDist<Ndim>;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */
    
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TRUNCATEDMULTIVARIATENORMALDIST_H */
