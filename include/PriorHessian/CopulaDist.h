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

//CopulaDistImpl namespace hides the definition of CopulaDist so that the wrapper in prior_hessian namspace can chec
//that only UnivariateDists with adaptable bounds are used as Marginal Dists.
namespace CopulaDistImpl { 
    
template<template <int> class CopulaTemplate, class... MarginalDistTs> 
class CopulaDist : public MultivariateDist {
    using IndexT = std::index_sequence_for<MarginalDistTs...>;
    static constexpr IdxT _num_dim =  sizeof...(MarginalDistTs);
    static constexpr IdxT _num_params = CopulaTemplate<_num_dim>::num_params() + meta::sum_in_order({MarginalDistTs::num_params()...});
public:
    using NdimVecT = arma::Col<double>::fixed<_num_dim>;
    using NdimMatT = arma::Mat<double>::fixed<_num_dim,_num_dim>;
    using NparamsVecT = arma::Col<double>::fixed<_num_params>;

    using MarginalDistTupleT = std::tuple<MarginalDistTs...>;    
    using CopulaT = CopulaTemplate<_num_dim>;
    
    template<size_t I>
    using MarginalDistT = std::tuple_element<I,MarginalDistTupleT>;
    
    static constexpr IdxT num_components() { return _num_dim; }
    static constexpr IdxT num_dim() { return _num_dim; }
    static constexpr IdxT num_params() {return _num_params;}
    
    template<class Vec>
    static bool check_params(const Vec &params);
    static bool check_copula_theta(double theta);
    
    static const StringVecT& param_names();
    static const NparamsVecT& param_lbound();
    static const NparamsVecT& param_ubound();
    static const NdimVecT& global_lbound();
    static const NdimVecT& global_ubound();
    
    CopulaDist();

    template<class Copula, class... DistTs,std::enable_if_t<sizeof...(DistTs)==sizeof...(MarginalDistTs),bool> Enable =true>
    CopulaDist(Copula&& copula, DistTs&&... dists);

    void initialize_copula(const CopulaT &_copula);
    void initialize_marginals(const MarginalDistTupleT &dists);

    NdimVecT lbound() const { return _lbound; }
    NdimVecT ubound() const { return _ubound; }
    
    template<class Vec, class Vec2>
    void set_bounds(const Vec& lbound, const Vec2& ubound);
        
    template<class Vec>
    void set_lbound(const Vec& lbound);

    template<class Vec>
    void set_ubound(const Vec& ubound);
        
    bool operator==(const CopulaDist<CopulaTemplate,MarginalDistTs...> &o) const;
    bool operator!=(const CopulaDist<CopulaTemplate,MarginalDistTs...> &o) const { return !this->operator==(o);}
    
    double get_param(int idx) const;
    NparamsVecT params() const;
    double get_copula_theta() const;
    void set_copula_theta(double theta);
    
    
    template<class Vec>
    void set_params(const Vec &params);    
    
//     NdimVecT mean() const;
//     NdimVecT mode() const;
    
    template<class Vec> double cdf(Vec x) const;
    template<class Vec> double pdf(const Vec &x) const;
    template<class Vec> double llh(const Vec &x) const;
    template<class Vec> double rllh(const Vec &x) const;
    template<class Vec> NdimVecT grad(const Vec &x) const;
    template<class Vec> NdimVecT grad2(const Vec &x) const;
    template<class Vec> NdimMatT hess(const Vec &x) const;
    
    template<class Vec,class Vec2>
    void grad_grad2_accumulate(const Vec &x, Vec2 &g, Vec2 &g2) const;
    template<class Vec,class Vec2,class Mat>
    void grad_hess_accumulate(const Vec &x, Vec2 &g, Mat &hess) const;
    
    template<class RngT>
    NdimVecT sample(RngT &rng) const;


     /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);   
    
    template<class IterT>
    void set_params_iter(IterT &params);
    
protected:
    static void check_bounds(double lbound, double ubound);
    
private:
    
    static StringVecT _param_names; //Cannonical names for parameters
    static NparamsVecT _param_lbound; //Lower bound on valid parameter values 
    static NparamsVecT _param_ubound; //Upper bound on valid parameter values
    static NdimVecT _global_lbound;
    static NdimVecT _global_ubound;

    //Static initializers are called only once.
    static bool initialize_param_names(); 
    template<std::size_t... I>
    static bool initialize_param_lbound(std::index_sequence<I...>);
    template<std::size_t... I>
    static bool initialize_param_ubound(std::index_sequence<I...>);
    template<std::size_t... I>
    static bool initialize_global_lbound(std::index_sequence<I...>);
    template<std::size_t... I>
    static bool initialize_global_ubound(std::index_sequence<I...>);
    template<std::size_t... I>
    static StringVecT marginal_param_names(std::index_sequence<I...>);
    template<std::size_t... I>
    static NdimVecT marginal_num_params(std::index_sequence<I...>);
    template<std::size_t... I>
    NdimVecT marginal_lbound(std::index_sequence<I...>) const; 
    template<std::size_t... I>
    NdimVecT marginal_ubound(std::index_sequence<I...>) const ;

    CopulaT copula;
    MarginalDistTupleT marginal_dists;
    NdimVecT _lbound;
    NdimVecT _ubound;
    
//     template<std::size_t... I>
//     NdimVecT initialize_global_lbound(std::index_sequence<I...>) const
//     { return {std::get<I>(marginal_dists).global_lbound()...}; }
//     
//     template<std::size_t... I>
//     NdimVecT initialize_global_ubound(std::index_sequence<I...>) const
//     { return {std::get<I>(marginal_dists).global_ubound()...}; }
// /
//     VecT
    
    template<std::size_t... I>
    NdimVecT initialize_lbound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).lbound()...}; }

    template<std::size_t... I>
    NdimVecT initialize_ubound(std::index_sequence<I...>) const
    { return {std::get<I>(marginal_dists).ubound()...}; }
    
};

} /* namespace prior_hessian::CopulaDistImpl */


template<template <int> class CopulaTemplate, class... MarginalDistTs>
using CopulaDist = CopulaDistImpl::CopulaDist<CopulaTemplate, BoundsAdaptedDistT<MarginalDistTs>...>;



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
    struct dist_adaptor_traits<CopulaDistImpl::CopulaDist<CopulaTemplate, DistTs...>> 
    {
        using bounds_adapted_dist = CopulaDistImpl::CopulaDist<CopulaTemplate, DistTs...>;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */



/* templated methods of CopulaDistImpl::CopulaDist */
namespace CopulaDistImpl {

/* helpers to expose the internal typenames */
// template<template <int> class CopulaTemplate, class... MarginalDistTs>
// using NdimVecT = typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT;
// 
// template<template <int> class CopulaTemplate, class... MarginalDistTs>
// using NparamsVecT = typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT;

/* static memeber variables */
template<template <int> class CopulaTemplate, class... MarginalDistTs>
StringVecT CopulaDist<CopulaTemplate,MarginalDistTs...>::_param_names; //Cannonical names for parameters

template<template <int> class CopulaTemplate, class... MarginalDistTs>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::_param_lbound; //Lower bound on valid parameter values 

template<template <int> class CopulaTemplate, class... MarginalDistTs>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::_param_ubound; //Upper bound on valid parameter values

template<template <int> class CopulaTemplate, class... MarginalDistTs>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::_global_lbound;

template<template <int> class CopulaTemplate, class... MarginalDistTs>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::_global_ubound;
    
/* Constructors */
template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Copula, class... DistTs, std::enable_if_t<sizeof...(DistTs)==sizeof...(MarginalDistTs),bool> Enable > //to allow universal-references.
CopulaDist<CopulaTemplate,MarginalDistTs...>::CopulaDist(Copula&& copula, DistTs&&... dists) 
    : MultivariateDist(),        
        copula(std::forward<Copula>(copula)),
        marginal_dists(make_adapted_bounded_dist(std::forward<DistTs>(dists))...),
        _lbound({dists.lbound()... }),
        _ubound({dists.ubound()... })        
{ }
               
template<template <int> class CopulaTemplate, class... MarginalDistTs>
CopulaDist<CopulaTemplate,MarginalDistTs...>::CopulaDist()
    : MultivariateDist(),
      copula(),
      marginal_dists(),
      _lbound(global_lbound()),
      _ubound(global_ubound())
{ 
}
    
template<template <int> class CopulaTemplate, class... MarginalDistTs>
void 
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_copula(const CopulaT &_copula)
{
    copula = _copula;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
void 
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_marginals(const MarginalDistTupleT &dists)
{
    marginal_dists = dists;
    _lbound = marginal_lbound(IndexT{});
    _ubound = marginal_ubound(IndexT{});
}    

template<template <int> class CopulaTemplate, class... MarginalDistTs>
const StringVecT& 
CopulaDist<CopulaTemplate,MarginalDistTs...>::param_names()
{ 
    static bool _dummy = initialize_param_names(); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_names; 
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
const typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT& 
CopulaDist<CopulaTemplate,MarginalDistTs...>::param_lbound()
{
    static bool _dummy = initialize_param_lbound(IndexT{}); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_lbound;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
const typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT& 
CopulaDist<CopulaTemplate,MarginalDistTs...>::param_ubound()
{
    static bool _dummy = initialize_param_ubound(IndexT{}); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_ubound;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
const typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT& 
CopulaDist<CopulaTemplate,MarginalDistTs...>::global_lbound()
{
    static bool _dummy = initialize_global_lbound(IndexT{}); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _global_lbound;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
const typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT& 
CopulaDist<CopulaTemplate,MarginalDistTs...>::global_ubound()
{
    static bool _dummy = initialize_global_ubound(IndexT{}); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _global_ubound;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
bool
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_param_names()
{
    _param_names.reserve(num_params());
    _param_names.emplace_back("copula_theta");
    auto mparam_names = marginal_param_names(IndexT{});
    auto nparams = marginal_num_params(IndexT{});
    int n=0;
    for(IdxT i=0;i<num_components(); i++) {
        std::ostringstream dim_name;
        dim_name<<"dim"<<i+1<<"_";
        for(IdxT k=0; k<nparams(i); k++) _param_names.emplace_back(dim_name.str()+mparam_names[n++]);
    }
    return true;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
bool
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_param_lbound(std::index_sequence<I...>)
{
    auto iter = _param_lbound.begin();
    *iter++ = CopulaT::param_lbound();
    meta::call_in_order({(iter = std::copy_n(MarginalDistT<I>::param_lbound().begin(),MarginalDistT<I>::num_params(),iter), 0)... });
    return true;    
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
bool
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_param_ubound(std::index_sequence<I...>)
{
    auto iter = _param_ubound.begin();
    *iter++ = CopulaT::param_ubound();
    meta::call_in_order({(iter = std::copy_n(MarginalDistT<I>::param_ubound().begin(),MarginalDistT<I>::num_params(),iter), 0)... });
    return true;    
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
bool
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_global_lbound(std::index_sequence<I...>)
{
//     _global_lbound = {MarginalDistT<I>::global_lbound()...};
    return true;    
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
bool
CopulaDist<CopulaTemplate,MarginalDistTs...>::initialize_global_ubound(std::index_sequence<I...>)
{
//     _global_ubound = {MarginalDistT<I>::global_ubound()...};
    return true;    
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
StringVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_param_names(std::index_sequence<I...>)
{
    StringVecT names;
    names.reserve(num_params());    
    auto iter = names.begin();
    meta::call_in_order({(iter = std::copy_n(MarginalDistT<I>::param_names().begin(),MarginalDistT<I>::num_params(),iter), 0)... });
    return names;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_num_params(std::index_sequence<I...>)
{
    return { MarginalDistT<I>::num_params()... };
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_lbound(std::index_sequence<I...>) const
{
    return { std::get<I>(marginal_dists).lbound()... };
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_ubound(std::index_sequence<I...>) const
{
    return { std::get<I>(marginal_dists).ubound()... };
}

    
} /* namepsace prior_hessian::CopulaDistImpl */
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_COPULADIST_H */
