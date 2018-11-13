/** @file CopulaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief CopulaDist base class.
 */
#ifndef PRIOR_HESSIAN_COPULADIST_H
#define PRIOR_HESSIAN_COPULADIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/MultivariateDist.h"
#include "PriorHessian/BoundsAdaptedDist.h"

namespace prior_hessian {

namespace Impl{
    
template<template <int> class CopulaTemplate, class... MarginalDistTs> 
//          meta::ConstructableIfIsSuperClassForAllT<UnivariateDist,DistTs...>=true,
//          meta::ConstructableIfAllDistsAreBoundedT<DistTs...>=true >
class CopulaDist : public MultivariateDist<sizeof...(MarginalDistTs)> {
    using IndexT = std::index_sequence_for<MarginalDistTs...>;
    static constexpr IdxT _num_dim =  sizeof...(MarginalDistTs);
    static constexpr IdxT _num_params = CopulaTemplate<_num_dim>::num_params() + meta::sum_in_order({MarginalDistTs::num_params()...});
public:
    static constexpr IdxT num_dim() { return _num_dim; }
    static constexpr IdxT num_components() { return _num_dim; }
    using MarginalDistTupleT = std::tuple<MarginalDistTs...>;    
    using CopulaT = CopulaTemplate<num_dim()>;
    using MultivariateDistBaseT = MultivariateDist<_num_dim>;
    using typename MultivariateDistBaseT::NdimVecT;
    using typename MultivariateDistBaseT::NdimMatT;
    
    static constexpr IdxT num_params() {return _num_params;}
    using NparamsVecT = arma::Col<double>::fixed<num_params()>;

            
    //This version enables initialization with non-bounded distributions via delegating constructor
    //Note the "Not" in the the 4th template argument.
//     template<class... DistTs>
// //          meta::ConstructableIfIsSuperClassForAllT<UnivariateDist,DistTs...>=true,
// //          meta::ConstructableIfNotAllDistsAreBoundedT<DistTs...>=true>
//     CopulaDist(CopulaT &&copula, DistTs&&... dists) : 
//         CopulaDist(copula,make_adapted_bounded_dist(std::forward<DistTs>(dists))...) {}
// 
//     template<class... DistTs>
// //          meta::ConstructableIfIsSuperClassForAllT<UnivariateDist,DistTs...>=true,
// //          meta::ConstructableIfAllDistsAreBoundedT<DistTs...>=true>
//     CopulaDist(const CopulaT &copula, DistTs&&... dists) 
//         : MultivariateDistBaseT({dists.lbound()...}, {dists.ubound()...}),        
//           copula(copula),
//           marginal_dists(make_adapted_bounded_dist_tuple(dists...)),
//           _global_lbound({dists.global_lbound()...}),
//           _global_ubound({dists.global_ubound()...})                 
//         { }
    
    CopulaDist() 
        : _global_lbound(initialize_global_lbound(IndexT())),
          _global_ubound(initialize_global_ubound(IndexT()))
    { 
        this->initialize_bounds(_global_lbound, _global_ubound);
    }
        
    void initialize_copula(const CopulaT &_copula)
    {
        copula = _copula;
    }
    
    void initialize_marginals(const MarginalDistTupleT &dists)
    {
        marginal_dists = dists;
        _global_lbound = initialize_global_lbound(IndexT{});
        _global_ubound = initialize_global_ubound(IndexT{});
        this->initialize_bounds(initialize_lbound(IndexT{}), initialize_ubound(IndexT{}));
    }    
//     template<class Vec, meta::EnableIfIsNotTupleAndIsNotSelfT<CopulaDist<Copulatemplate,DistTs...>,Vec>>
//     CopulaDist(const Vec& params);
//     
//     template<class Vec, class Vec2>
//     CopulaDist(const Vec& params, Vec2&& lbound, Vec2&& ubound);
//     
//     void set_bounds(double lbound, double ubound);
//     void set_lbound(double lbound);
//     void set_ubound(double ubound);
    double global_lbound() const { return _global_lbound; }
    double global_ubound() const { return _global_ubound; }
protected:
    static void check_bounds(double lbound, double ubound);
    
private:
    CopulaT copula;
    MarginalDistTupleT marginal_dists;
    NdimVecT _global_lbound;
    NdimVecT _global_ubound;
    
    template<std::size_t... I>
    NdimVecT initialize_global_lbound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).global_lbound()...}; }
    
    template<std::size_t... I>
    NdimVecT initialize_global_ubound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).global_ubound()...}; }
    
    template<std::size_t... I>
    NdimVecT initialize_lbound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).lbound()...}; }

    template<std::size_t... I>
    NdimVecT initialize_ubound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).ubound()...}; }
    
};

} /* namespace prior_hessian::(annon) */


template<template <int> class CopulaTemplate, class... MarginalDistTs>
using CopulaDist = Impl::CopulaDist<CopulaTemplate, BoundsAdaptedDistT<MarginalDistTs>...>;



template<template <int> class CopulaTemplate, class... MarginalDistTs>
CopulaDist<CopulaTemplate,MarginalDistTs...>
make_copula_dist(CopulaTemplate<sizeof...(MarginalDistTs)>&& copula, MarginalDistTs&&... dists)
{
    return {std::forward<CopulaTemplate<sizeof...(MarginalDistTs)>>(copula), 
                make_adapted_bounded_dist(std::forward<MarginalDistTs>(dists))...};
}


namespace detail
{
    /* Indicate that all CopulaDists have adaptable bounds by default */
//     template<class CopulaT>
//     struct dist_adaptor_traits<CopulaT> 
//     {
//         using bounds_adapted_dist = CopulaT;
//         static constexpr std::ConstructableIfIsCopulaT<CopulaDist, CopulaT> adaptable_bounds = true;
//     };
    
    template<template <int> class CopulaTemplate, class... DistTs>
    struct dist_adaptor_traits<Impl::CopulaDist<CopulaTemplate, DistTs...>> 
    {
        using bounds_adapted_dist = Impl::CopulaDist<CopulaTemplate, DistTs...>;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */


} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_COPULADIST_H */
