/** @file BoundsAdaptedDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Functions and type-traits to enable easy wrapping of distributions in bounds-modifiable adapters.
 * 
 * The bounds-adapted version of a distribution is a distribution that has been wrapped by an adapter-class
 * that modifies the distribution so that the bounds can be set to finite values.
 * 
 * The main types of adapters are
 *    - TruncatedDist:
 *      - Has global_ubound and global_lbound which may both be infinite (e.g., with the NormalDist)
 *      - Adapts distributions: NormalDist, GammaDist
 *    - UpperTrunctedDist: Adapts distributions which already have an inherit lower-bound as a parameter
 *      - Adapts distributions: ParatoDist
 *    - ScaledDist:
 *      - Adapts distributions with finite domain like the Beta distribution by scaling them to 
 *          arbitrary finite bounds
 *      - Adapts distributions: BetaDist, SymmetricBetaDist
 * 
 * Functions:
 *   make_adapted_bounded_dist
 *      - Make a bounds-adapted version of given distribution
 *   make_adapted_bounded_dist_tuple -
 *      - Make a tuple of bounds-adapted version of given distributions
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_BOUNDSADAPTEDDIST_H
#define PRIOR_HESSIAN_BOUNDSADAPTEDDIST_H

#include <utility>
#include <tuple>
#include <type_traits>

#include "PriorHessian/Meta.h"

namespace prior_hessian {

namespace detail 
{
    template<class Dist>
    class dist_adaptor_traits {
    public:
        using bounds_adapted_dist = void;
        static constexpr bool adaptable_bounds = false;
    };
    
    
    /* Forward declaration of class template specializations for each distribution.
     * TODO: Can this uglyness be eliminated/automated
     */
    class NormalDist;
    template<> class dist_adaptor_traits<NormalDist>;
    class GammaDist;
    template<> class dist_adaptor_traits<GammaDist>;
    class ParetoDist;
    template<> class dist_adaptor_traits<ParetoDist>;
    class SymmetricBetaDist;
    template<> class dist_adaptor_traits<SymmetricBetaDist>;
    
    /** Type traits class for distribution type DistT.  
    *
    * The traits class describes the Adaptor classes applicable to each individual distribution
    */
    template<class DistT> using DistTraitsT = detail::dist_adaptor_traits<std::decay_t<DistT>>;
}

/** The bounds-adapted distribution type for a given distribution type DistT
 * This is the adapted version of the class, i.e., the class that allows truncation or scaling so that the lower and upper bounds are settable.
 */
template<class DistT> using BoundsAdaptedDistT = 
    typename detail::dist_adaptor_traits<std::decay_t<DistT>>::bounds_adapted_dist;

/* Helpers for make_adapted_bounded_dist() */
namespace detail 
{
    template<class... Ts,std::size_t... I>
    std::tuple<BoundsAdaptedDistT<Ts>...>
    make_adapted_bounded_dist_tuple(std::tuple<Ts...>&& dists, std::index_sequence<I...> )
    {
        return std::make_tuple(make_adapted_bounded_dist(std::get<I>(std::move(dists)))...);
    }

    template<class... Ts,std::size_t... I>
    std::tuple<BoundsAdaptedDistT<Ts>...>
    make_adapted_bounded_dist_tuple(const std::tuple<Ts...>& dists, std::index_sequence<I...> )
    {
        return std::make_tuple(make_adapted_bounded_dist(std::get<I>(dists))...);
    }
} /* namespace detail */

/** make_adapted_bounded_dist() [4-forms]
 * If the given distribution is not bounded then the appropriate bounding distribution is wrapped arround it.
 * We detect the boundedness of the distribution using detail::adaptable_bounds type-traits class.
 * Can be replaced with constexpr if in c++17.  Uses SFINAE in c++14.
 */
template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
std::enable_if_t< detail::DistTraitsT<Dist>::adaptable_bounds, Dist>
make_adapted_bounded_dist(Dist &&dist)
{ return dist; }

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
std::enable_if_t< !detail::DistTraitsT<Dist>::adaptable_bounds, BoundsAdaptedDistT<Dist>>
make_adapted_bounded_dist(Dist &&dist)
{ return {std::forward<Dist>(dist)}; }

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
std::enable_if_t<detail::DistTraitsT<Dist>::adaptable_bounds, Dist>
make_adapted_bounded_dist(Dist &&dist, double lbound, double ubound)
{ 
    dist.set_bounds(lbound,ubound);
    return dist; 
}

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
std::enable_if_t<!detail::DistTraitsT<Dist>::adaptable_bounds, BoundsAdaptedDistT<Dist>>
make_adapted_bounded_dist(Dist &&dist, double lbound, double ubound)
{ 
    return {std::forward<Dist>(dist),lbound,ubound}; 
}


/* make_adapted_bounded_dist_tuple(...) [3-forms]
 * Accepts variadic set of distributions or a tuple of distributions.  Each distribution can be already bounds adapted
 * or not. Wraps, each unwraped distribution in the appropriate BoundsAdaptedDistT.
 * Uses perfect-forwarding.
 * 
 */
template<class... Ts>
std::tuple<BoundsAdaptedDistT<Ts>...>
make_adapted_bounded_dist_tuple(Ts&&... ts) 
{
    return std::make_tuple(make_adapted_bounded_dist(std::forward<Ts>(ts))...);
}

template<class... Ts>
std::tuple<BoundsAdaptedDistT<Ts>...>
make_adapted_bounded_dist_tuple(std::tuple<Ts...>&& dists) 
{
    return detail::make_adapted_bounded_dist_tuple(std::move(dists), std::index_sequence_for<Ts...>{});
}

template<class... Ts>
std::tuple<BoundsAdaptedDistT<Ts>...>
make_adapted_bounded_dist_tuple(const std::tuple<Ts...>& dists) 
{
    return detail::make_adapted_bounded_dist_tuple(dists, std::index_sequence_for<Ts...>{});
}
    

} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_BOUNDS_ADAPTED_DIST_H */
