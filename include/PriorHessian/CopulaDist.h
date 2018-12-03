/** @file CopulaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief CopulaDist base class.
 */
#ifndef PRIOR_HESSIAN_COPULADIST_H
#define PRIOR_HESSIAN_COPULADIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/Meta.h"
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
    
    NparamsVecT params() const;
    double get_copula_theta() const;
    void set_copula_theta(double theta);
    template<class Vec>
    void set_params(const Vec &params);    
    
//     NdimVecT mean() const;
//     NdimVecT mode() const;
    
    template<class Vec> double cdf(const Vec &x) const;
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
    template<class IterT, std::size_t... I>
    static bool check_marginal_params(IterT &params_it, std::index_sequence<I...>);
    
    template<std::size_t... I>
    NdimVecT marginal_lbound(std::index_sequence<I...>) const; 
    template<std::size_t... I>
    NdimVecT marginal_ubound(std::index_sequence<I...>) const ;
    template<class Iter, std::size_t... I>
    void set_marginal_bounds(Iter &lb_it, Iter &ub_it, std::index_sequence<I...>) const;
    template<class Iter, std::size_t... I>
    void set_marginal_lbound(Iter &lb_it, std::index_sequence<I...>) const;
    template<class Iter, std::size_t... I>
    void set_marginal_ubound(Iter &ub_it, std::index_sequence<I...>) const;
    template<class Iter, std::size_t... I>
    void append_marginal_params(Iter &it, std::index_sequence<I...>) const;
    template<class Iter,std::size_t... I>
    void set_marginal_params(Iter &it, std::index_sequence<I...>) const;
    
    
    template<class InIter,class OutIter, std::size_t... I>
    void compute_marginal_cdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const;
    template<class InIter,class OutIter, std::size_t... I>
    void compute_marginal_pdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const;
    template<class InIter,class OutIter, std::size_t... I>
    void compute_marginal_icdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const;
    
    template<class OCopulaT, std::size_t... I>
    bool marginals_are_equal(const OCopulaT &o,std::index_sequence<I...>) const; //helper for operator=()

    /* private member variables */
    CopulaT copula;
    MarginalDistTupleT marginals;
    NdimVecT _lbound;
    NdimVecT _ubound;
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
    template<template <int> class CopulaTemplate, class... DistTs>
    struct dist_adaptor_traits<CopulaDistImpl::CopulaDist<CopulaTemplate, DistTs...>> 
    {
        using bounds_adapted_dist = CopulaDistImpl::CopulaDist<CopulaTemplate, DistTs...>;
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace prior_hessian::detail */



/* templated methods of CopulaDistImpl::CopulaDist */
namespace CopulaDistImpl {

/* static memeber variable definition */
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
        marginals(make_adapted_bounded_dist(std::forward<DistTs>(dists))...),
        _lbound({dists.lbound()... }),
        _ubound({dists.ubound()... })        
{ }
               
template<template <int> class CopulaTemplate, class... MarginalDistTs>
CopulaDist<CopulaTemplate,MarginalDistTs...>::CopulaDist()
    : MultivariateDist(),
      copula(),
      marginals(),
      _lbound(global_lbound()),
      _ubound(global_ubound())
{ 
}


/* public member functions */
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
    marginals = dists;
    _lbound = marginal_lbound(IndexT{});
    _ubound = marginal_ubound(IndexT{});
}    

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec, class Vec2>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_bounds(const Vec& new_lbound, const Vec2& new_ubound)
{
    if( !arma::all(  new_lbound >= global_lbound()) ) {   //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid lbound:"<<new_lbound.t()<<" with regard to global_lbound:"<<global_lbound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all( new_ubound <= global_ubound()) ) {    //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<new_ubound.t()<<" with regard to global_ubound:"<<global_ubound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all( new_lbound < new_ubound) ) { 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid bounds lbound:"<<new_lbound.t()<<" >= ubound:"<<new_ubound.t();
        throw ParameterValueError(msg.str());
    }
    _lbound = new_lbound;
    _ubound = new_ubound;
    set_marginal_bounds(_lbound.begin(), _ubound.begin(), IndexT{}); //Update marginal bounds
}
    
template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_lbound(const Vec& new_lbound)
{
    if( !arma::all( new_lbound >= global_lbound()) ) {   //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_lbound: Invalid lbound:"<<new_lbound.t()<<" with regard to global_lbound:"<<global_lbound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all( new_lbound < ubound()) ) { 
        std::ostringstream msg;
        msg<<"set_lbound: Invalid bounds. new lbound:"<<new_lbound.t()<<" >= ubound:"<<ubound().t();
        throw ParameterValueError(msg.str());
    }
    _lbound = new_lbound;
    set_marginal_lbound(_lbound.begin(), IndexT{}); //Update marginal bounds
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_ubound(const Vec& new_ubound)
{
    if( !arma::all( new_ubound <= global_ubound()) ) {    //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_ubound: Invalid ubound:"<<new_ubound.t()<<" with regard to global_ubound:"<<global_ubound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all( lbound() < new_ubound) ) { 
        std::ostringstream msg;
        msg<<"set_ubound: Invalid bounds lbound:"<<lbound().t()<<" >= new ubound:"<<new_ubound.t();
        throw ParameterValueError(msg.str());
    }
    _ubound = new_ubound;
    set_marginal_ubound(_ubound.begin(), IndexT{}); //Update marginal bounds
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
bool CopulaDist<CopulaTemplate,MarginalDistTs...>::operator==(const CopulaDist<CopulaTemplate,MarginalDistTs...> &o) const
{
    return copula == o.copula && marginals_are_equal(o,IndexT{});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NparamsVecT 
CopulaDist<CopulaTemplate,MarginalDistTs...>::params() const
{
    NparamsVecT p;
    auto it = p.begin();
    append_marginal_params(it,IndexT{});
    return p;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
double CopulaDist<CopulaTemplate,MarginalDistTs...>::get_copula_theta() const
{
    return copula.theta();
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_copula_theta(double theta)
{
    copula.set_theta(theta);
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_params(const Vec &params)
{
    auto it = params.begin();
    copula.set_theta(*it++);
    set_marginal_params(it,IndexT{});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class IterT>
bool CopulaDist<CopulaTemplate,MarginalDistTs...>::check_params_iter(IterT &params_it)
{
    return check_params_iter(params_it) && marginal_check_params(params_it,IndexT{});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class IterT>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_params_iter(IterT &params_it)
{
    copula.set_theta(*params_it++);
    marginal_set_params(params_it,IndexT{});
}

/* public computational member functions */

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
double CopulaDist<CopulaTemplate,MarginalDistTs...>::cdf(const Vec& x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    return copula.cdf(marginal_cdf);
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
double CopulaDist<CopulaTemplate,MarginalDistTs...>::pdf(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_pdf;
    compute_marginal_pdf(x.begin(),marginal_pdf.begin(),IndexT{});
    return copula.pdf(marginal_cdf) * arma::prod(marginal_pdf);
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
double CopulaDist<CopulaTemplate,MarginalDistTs...>::llh(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_llh;
    compute_marginal_llh(x.begin(),marginal_llh.begin(),IndexT{});
    return copula.llh(marginal_cdf) + arma::sum(marginal_llh);
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
double CopulaDist<CopulaTemplate,MarginalDistTs...>::rllh(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_rllh;
    compute_marginal_rllh(x.begin(),marginal_rllh.begin(),IndexT{});
    return copula.rllh(marginal_cdf) + arma::sum(marginal_rllh);
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT 
CopulaDist<CopulaTemplate,MarginalDistTs...>::grad(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_grad;
    compute_marginal_grad(x.begin(),marginal_grad.begin(),IndexT{});
    return copula.grad(marginal_cdf) + marginal_grad;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT 
CopulaDist<CopulaTemplate,MarginalDistTs...>::grad2(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_grad2;
    compute_marginal_grad2(x.begin(),marginal_grad2.begin(),IndexT{});
    return copula.grad2(marginal_cdf) + marginal_grad2;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec> 
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimMatT 
CopulaDist<CopulaTemplate,MarginalDistTs...>::hess(const Vec &x) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_grad2;
    compute_marginal_grad2(x.begin(),marginal_grad2.begin(),IndexT{});
    auto H = copula.hess(marginal_cdf);
    H.diag() += marginal_grad2;
    return H;
}


template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec,class Vec2>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::grad_grad2_accumulate(const Vec &x, Vec2 &g, Vec2 &g2) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_grad;
    compute_marginal_grad(x.begin(),marginal_grad.begin(),IndexT{});
    NdimVecT marginal_grad2;
    compute_marginal_grad2(x.begin(),marginal_grad2.begin(),IndexT{});
    g += copula.grad(marginal_cdf) + marginal_grad;
    g2 += copula.grad2(marginal_cdf) + marginal_grad2;
}


template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Vec,class Vec2,class Mat>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::grad_hess_accumulate(const Vec &x, Vec2 &g, Mat &hess) const
{
    NdimVecT marginal_cdf;
    compute_marginal_cdf(x.begin(),marginal_cdf.begin(),IndexT{});
    NdimVecT marginal_grad;
    compute_marginal_grad(x.begin(),marginal_grad.begin(),IndexT{});
    NdimVecT marginal_grad2;
    compute_marginal_grad2(x.begin(),marginal_grad2.begin(),IndexT{});

    g += copula.grad(marginal_cdf) + marginal_grad;
    auto H = copula.hess(marginal_cdf);
    H.diag() += marginal_grad2;
    hess += H;
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class RngT>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT 
CopulaDist<CopulaTemplate,MarginalDistTs...>::sample(RngT &rng) const
{
    auto u = copula.sample(rng);
    compute_marginal_icdf(u.begin(),u.begin(),IndexT{});
    return u;
}

/* public static member functions */
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

/* private static member functions */
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
    meta::call_in_order({(iter = std::copy_n(MarginalDistT<I>::param_names().begin(),MarginalDistT<I>::num_params_static(),iter), 0)... });
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
template<class IterT, std::size_t... I>
bool CopulaDist<CopulaTemplate,MarginalDistTs...>::check_marginal_params(IterT &params_it, std::index_sequence<I...>)
{
    return meta::logical_and_in_order({MarginalDistT<I>::check_params_iter(params_it)...});
}
    
/* Private non-static member functions */
template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_lbound(std::index_sequence<I...>) const
{
    return { std::get<I>(marginals).lbound()... };
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<std::size_t... I>
typename CopulaDist<CopulaTemplate,MarginalDistTs...>::NdimVecT
CopulaDist<CopulaTemplate,MarginalDistTs...>::marginal_ubound(std::index_sequence<I...>) const
{
    return { std::get<I>(marginals).ubound()... };
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Iter, std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_marginal_bounds(Iter &lb_it,Iter &ub_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(std::get<I>(marginals).set_bounds(*lb_it++,*ub_it++),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Iter, std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_marginal_lbound(Iter &lb_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(std::get<I>(marginals).set_lbound(*lb_it++),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Iter, std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_marginal_ubound(Iter &ub_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(std::get<I>(marginals).set_ubound(*ub_it++),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Iter, std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::append_marginal_params(Iter &it, std::index_sequence<I...>) const
{
    meta::call_in_order({(std::get<I>(marginals).append_params_iter(it),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class Iter, std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::set_marginal_params(Iter &it, std::index_sequence<I...>) const
{
    meta::call_in_order({(std::get<I>(marginals).set_params_iter(it),0)...});    
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class OCopulaT, std::size_t... I>
bool CopulaDist<CopulaTemplate,MarginalDistTs...>::marginals_are_equal(const OCopulaT &o, std::index_sequence<I...>) const
{
    return meta::logical_and_in_order({std::get<I>(marginals) == std::get<I>(o.marginals)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class InIter,class OutIter,std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::compute_marginal_cdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(*out_it++ = std::get<I>(marginals).cdf(*in_it++),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class InIter,class OutIter,std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::compute_marginal_pdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(*out_it++ = std::get<I>(marginals).pdf(*in_it++),0)...});
}

template<template <int> class CopulaTemplate, class... MarginalDistTs>
template<class InIter,class OutIter,std::size_t... I>
void CopulaDist<CopulaTemplate,MarginalDistTs...>::compute_marginal_icdf(InIter in_it, OutIter out_it, std::index_sequence<I...>) const
{
    meta::call_in_order({(*out_it++ = std::get<I>(marginals).icdf(*in_it++),0)...});
}


    
} /* namepsace prior_hessian::CopulaDistImpl */
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_COPULADIST_H */
