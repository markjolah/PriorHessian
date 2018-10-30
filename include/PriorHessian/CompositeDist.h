/** @file CompositeDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief The Frank copula computations
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_COMPOSITEDIST_H
#define PRIOR_HESSIAN_COMPOSITEDIST_H

#include<utility>
#include<memory>
#include<unordered_map>
#include<unordered_set>
#include<sstream>

#include<armadillo>

#include "PriorHessian/Meta.h"
#include "PriorHessian/util.h"
#include "PriorHessian/PriorHessianError.h"
#include "PriorHessian/BoundsAdaptedDist.h"

#include "PriorHessian/AnyRng/AnyRng.h"

namespace prior_hessian {

class UnivariateDist;
template<int> class MultivariateDist;
/** @brief A probability distribution made of independent component distributions composing groups of 1 or more variables.
 * 
 * CompositeDist is a world unto itself.
 * 
 * class UnivariateDistInterface {
 *   static constexpr IdxT num_dim();
 *   static constexpr IdxT num_params();
 *   static const StringVecT param_names;
 *   double lbound() const; 
 *   double ubound() const;
 *   void set_bounds(double lbound, double ubound);
 *   void set_lbound(double lbound);
 *   void set_ubound(double ubound);
 *   double get_param(int idx) const;
 *   void set_param(int idx, double val);
 *   double cdf(double x) const;
 *   double icdf(double u) const;
 *   double pdf(double x) const;
 *   double llh(double x) const;
 *   double rllh(double x) const;
 *   double grad(double x) const;
 *   double grad2(double x) const;
 *   void grad_grad2_accumulate(double x, double &g, double &g2) const;
 *   template< class RngT > double sample(RngT &rng) const;
 * }
 * 
 * 
 */

class CompositeDist
{
public:
    template<class DistT, typename Enable=void>
    class ComponentDistAdaptor;
    
public:
    using AnyRngT = any_rng::AnyRng<std::size_t>;
    
    CompositeDist();
    
    /** @brief Initialize from a tuple of of UnivariateDist's or MulitvariateDist's */
    template<class... Ts, meta::ConstructableIfAllAreNotTupleAndAreNotSelfT<CompositeDist,Ts...> = true>
    explicit CompositeDist(Ts&&... dists)   
        : handle{ new DistTuple<ComponentDistT<Ts>...>{make_component_dist(std::forward<Ts>(dists))...} }
    { initialize_from_handle(); }
    
    template<class... Ts>
    explicit CompositeDist(std::tuple<Ts...>&& dist_tuple)
        : handle{ new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(std::move(dist_tuple))} }
    { initialize_from_handle(); }

    template<class... Ts>
    explicit CompositeDist(const std::tuple<Ts...>& dist_tuple)
        : handle{ new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(dist_tuple)} }
    { initialize_from_handle(); }
    
    void initialize() { clear(); }
    void initialize(const std::tuple<>&) { clear(); }
    void initialize(std::tuple<>&&) { clear(); }
    template<class... Ts, typename=meta::EnableIfAllAreNotTupleT<Ts...>>
    void initialize(Ts&&... dists)
    {
        handle = std::unique_ptr<DistTupleHandle>{ 
                    new DistTuple<ComponentDistT<Ts>...>{make_component_dist(std::forward<Ts>(dists))...} };
        initialize_from_handle();
    }

    template<class... Ts, typename=meta::EnableIfAllAreNotTupleT<Ts...>>
    void initialize(std::tuple<Ts...>&& dist_tuple)
    {
        handle = std::unique_ptr<DistTupleHandle>{ 
                    new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(std::move(dist_tuple))} };
        initialize_from_handle();
    }

    template<class... Ts, typename=meta::EnableIfNonEmpty<Ts...>>
    void initialize(const std::tuple<Ts...>& dist_tuple)
    {
        handle = std::unique_ptr<DistTupleHandle>{ 
                    new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(dist_tuple)} };
        initialize_from_handle();
    }

    
    CompositeDist(const CompositeDist &);
    CompositeDist& operator=(const CompositeDist &);     
    CompositeDist(CompositeDist&&);
    CompositeDist& operator=(CompositeDist&&);  

    void clear();
       
    template<class... Ts> 
    const std::tuple<Ts...>& get_dist_tuple() const; 
    
    bool is_empty() const { return handle->num_dists()==0; }
    operator bool() const { return handle->num_dists()>0; }
    IdxT num_components() const { return handle->num_dists(); }
    TypeInfoVecT component_types() const { return handle->component_types(); }
    const StringVecT& component_names() const;
    template<class StringVec> 
    void set_component_names(StringVec &&names);
    
    bool operator==(const CompositeDist &o) const;
    bool operator!=(const CompositeDist &o) const { return !this->operator==(o); }
    
    /* Dimensionality and variable names */
    IdxT num_dim() const { return handle->num_dim(); }
    UVecT num_dim_components() const { return handle->num_dim_components(); }
    const StringVecT& dim_variables() const;
    template<class StringVec> 
    void set_dim_variables(StringVec &&vars); 
    
    /* Bounds */
    VecT lbound() const { return handle->lbound(); }
    VecT ubound() const { return handle->ubound(); }
    VecT global_lbound() const { return handle->global_lbound(); }
    VecT global_ubound() const { return handle->global_ubound(); }
    bool in_bounds(const VecT &u) const { return arma::all(lbound()<=u) && arma::all(u<=ubound()); }
    /*Check all columns are in bounds vectors */
    bool in_bounds_all(const MatT &u) const 
    { 
        return arma::all(arma::min(u,1)>=lbound()) &&arma::all(arma::max(u,1)<=ubound());
    }
    
    void set_lbound(const VecT &new_bound) { handle->set_lbound(new_bound); }
    void set_ubound(const VecT &new_bound) { handle->set_ubound(new_bound); }
    void set_bounds(const VecT &new_lbound,const VecT &new_ubound) { handle->set_bounds(new_lbound, new_ubound); }

    /* Distribution Parameters */
    IdxT num_params() const { return handle->num_params(); }
    UVecT num_params_components() const { return handle->num_params_components(); }
    VecT params() const { return handle->params(); } 
    void set_params(const VecT &new_params) { handle->set_params(new_params); }
    bool check_params(const VecT &new_params) const { return handle->check_params(new_params); }
    VecT params_lbound() const { return handle->params_lbound(); }
    VecT params_ubound() const { return handle->params_ubound(); }
    std::vector<VecT> params_components() const { return handle->params_components(); } /* Seperate parameters int a VecT for each component */

    /* Distribution Parameter Descriptions (names) */
    const StringVecT& param_names() const;
    template<class StringVec> 
    void set_param_names(StringVec &&vars); 

    /* Convenience functions for working with parameters by name */
    bool has_param(const std::string &name) const;
    double get_param_value(const std::string &name) const; 
    int get_param_index(const std::string &name) const; 
    void set_param_value(const std::string &name, double value);
    
    //Functions mapped over underlying distributions
    double cdf(const VecT &u) const { return handle->cdf(u); }
    double pdf(const VecT &u) const { return handle->pdf(u); }
    double llh(const VecT &u) const { return handle->llh(u); }
    double rllh(const VecT &u) const { return handle->rllh(u); }
    VecT grad(const VecT &u) const
    {
        VecT g(num_dim(), arma::fill::zeros);
        handle->grad_accumulate(u,g);
        return g;
    }

    VecT grad2(const VecT &u) const
    {
        VecT g2(num_dim(), arma::fill::zeros);
        handle->grad2_accumulate(u,g2);
        return g2;
    }

    MatT hess(const VecT &u) const
    {
        MatT h(num_dim(),num_dim(),arma::fill::zeros);
        handle->hess_accumulate(u,h);
        return h;
    }
    
    /* Accumulate methods add thier contributions to respective grad, grad2 or hess arguments 
     * These are more efficient.  They use the Dist.xxx_acumulate methods, which are intern provided by the
     * CRTP UnivariateDist and MulitvariateDists base class templates.
     * 
     * All this technology should make these calls very close to custom coded grad and hess calls for any combination of dists.
     */
    void grad_accumulate(const VecT &theta, VecT &grad) const { return handle->grad_accumulate(theta,grad); }
    void grad2_accumulate(const VecT &theta, VecT &grad2) const { return handle->grad2_accumulate(theta,grad2); }
    void hess_accumulate(const VecT &theta, MatT &hess) const { return handle->hess_accumulate(theta,hess); }
    void grad_grad2_accumulate(const VecT &theta, VecT &grad, VecT &grad2) const { return handle->grad_grad2_accumulate(theta,grad,grad2); }
    void grad_hess_accumulate(const VecT &theta, VecT &grad, MatT &hess) const { return handle->grad_hess_accumulate(theta,grad,hess); }
    //Convenience methods for the lazy.
    VecT make_zero_grad() const { return {num_dim(),arma::fill::zeros}; }
    MatT make_zero_hess() const { return {num_dim(),num_dim(),arma::fill::zeros}; }
    
    VecT sample(AnyRngT &rng) { return handle->sample(rng); }
    MatT sample(AnyRngT &rng, IdxT num_samples) { return handle->sample(rng,num_samples); }

    template<class RngT>
    VecT sample(RngT &&rng) 
    { 
        AnyRngT anyrng{std::forward<RngT>(rng)};
        return handle->sample(anyrng); 
    }

    template<class RngT>
    MatT sample(RngT &&rng, IdxT num_samples) 
    { 
        AnyRngT anyrng{std::forward<RngT>(rng)};
        return handle->sample(anyrng,num_samples); 
    }

    /* Per-component values for debugging and plotting purposes */
    VecT llh_components(const VecT &u) const { return handle->llh_components(u); }
    VecT rllh_components(const VecT &u) const { return handle->rllh_components(u); }

private:
    
    /** @brief Model interface
     */
    class DistTupleHandle
    {
    public:
        virtual ~DistTupleHandle() = default;
        virtual std::unique_ptr<DistTupleHandle> clone() const = 0;
        virtual const std::type_info& type_info() const = 0;
        virtual bool is_equal(const DistTupleHandle &) const =0;
        virtual IdxT num_dists() const = 0;
        virtual TypeInfoVecT component_types() const = 0;
        virtual IdxT num_dim() const = 0;
        virtual UVecT num_dim_components() const = 0;
        virtual VecT lbound() const = 0;
        virtual VecT ubound() const = 0;
        virtual VecT global_lbound() const = 0;
        virtual VecT global_ubound() const = 0;
        virtual void set_lbound(const VecT &hew_lbound) = 0;
        virtual void set_ubound(const VecT &hew_ubound) = 0;
        virtual void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) = 0;
        virtual IdxT num_params() const = 0;
        virtual UVecT num_params_components() const = 0;
        virtual VecT params() const = 0;        
        virtual void set_params(const VecT &params) = 0;
        virtual bool check_params(const VecT &new_params) const = 0;
        virtual VecT params_lbound() const = 0;
        virtual VecT params_ubound() const = 0;
        virtual std::vector<VecT> params_components() const = 0;
        virtual StringVecT param_names() const = 0;
        virtual double cdf(const VecT &u) const = 0;
        virtual double pdf(const VecT &u) const = 0;
        virtual double llh(const VecT &u) const = 0;
        virtual double rllh(const VecT &u) const = 0;
        virtual void grad_accumulate(const VecT &u, VecT &grad) const = 0;
        virtual void grad2_accumulate(const VecT &u, VecT &grad2) const = 0;
        virtual void hess_accumulate(const VecT &u, MatT &hess) const = 0;
        virtual void grad_grad2_accumulate(const VecT &u, VecT &grad, VecT &grad2) const = 0;
        virtual void grad_hess_accumulate(const VecT &u, VecT &grad, MatT &hess) const = 0;
        virtual VecT sample(AnyRngT &rng) = 0;
        virtual MatT sample(AnyRngT &rng, IdxT nSamples) = 0;
        virtual VecT llh_components(const VecT &u) const = 0;
        virtual VecT rllh_components(const VecT &u) const = 0;
    }; /* class DistTupleHandle */
    
    template<class... Ts>
    class DistTuple : public DistTupleHandle
    {        
        using IndexT = std::index_sequence_for<Ts...>;
        constexpr static IdxT _num_dists = sizeof...(Ts);
        constexpr static IdxT _num_dim = meta::sum_in_order({Ts::num_dim()...});
        constexpr static IdxT _num_params = meta::sum_in_order({Ts::num_params()...});
        using StaticSizeArrayT = std::array<IdxT,_num_dists>;
        constexpr static StaticSizeArrayT _component_num_dim = {{Ts::num_dim()...}}; 
        constexpr static StaticSizeArrayT _component_num_params = {{Ts::num_params()...}}; 
    public:  
        explicit DistTuple(const std::tuple<Ts...> &_dists) : dists{_dists} { }
        explicit DistTuple(std::tuple<Ts...>&& _dists) : dists{std::move(_dists)} { }
        
        template<meta::ConstructableIfIsTemplateForAllT<ComponentDistAdaptor,Ts...> = true>
        explicit DistTuple(Ts&&... _dists) : dists{std::make_tuple(std::forward<Ts>(_dists)...)} { }
                
        const std::type_info& type_info() const override { return typeid(dists); }
        std::unique_ptr<DistTupleHandle> clone() const override { return std::make_unique<DistTuple<Ts...>>(dists); }
        bool is_equal(const DistTupleHandle &o) const override 
        {  return is_equal(static_cast<const DistTuple<Ts...> &>(o), IndexT{}); }
            
        IdxT num_dists() const override { return _num_dists; }
        IdxT num_dim() const override { return _num_dim; }
        IdxT num_params() const override { return _num_params; }
        UVecT num_dim_components() const override { return {Ts::num_dim()...}; }
        UVecT num_params_components() const override { return {Ts::num_params()...}; }
        TypeInfoVecT component_types() const override { return {std::type_index(typeid(Ts))...}; }

        VecT lbound() const override
        {
            VecT lb(_num_dim);
            append_lbound(lb.begin(),IndexT());
            return lb;
        }
        
        VecT ubound() const override
        {
            VecT ub(_num_dim);
            append_ubound(ub.begin(),IndexT());
            return ub;
        }
        
        VecT global_lbound() const override
        {
            VecT lb(_num_dim);
            append_global_lbound(lb.begin(),IndexT());
            return lb;
        }
        
        VecT global_ubound() const override
        {
            VecT ub(_num_dim);
            append_global_ubound(ub.begin(),IndexT());
            return ub;
        }
        
        void set_lbound(const VecT &new_bound) override { set_lbound(new_bound.begin(), IndexT{}); }
        void set_ubound(const VecT &new_bound) override { set_ubound(new_bound.begin(), IndexT{}); }
        void set_bounds(const VecT &new_lbound,const VecT &new_ubound) override { set_bounds(new_lbound.begin(), new_ubound.begin(), IndexT{}); }

        VecT params() const override
        {
            VecT params(_num_params);
            append_params(params.begin(),IndexT{});
            return params;
        }
        
        void set_params(const VecT &params) override { set_params(params.begin(),IndexT{}); }
        bool check_params(const VecT &new_params) const override { return check_params(new_params.begin(), IndexT{}); }
        
        VecT params_lbound() const override
        { 
            VecT lb(_num_params);
            append_params_lbound(lb.begin(),IndexT{});
            return lb;
        }

        VecT params_ubound() const override
        { 
            VecT lb(_num_params);
            append_params_ubound(lb.begin(),IndexT{});
            return lb;
        }

        std::vector<VecT> params_components() const override
        {
            return params_components(IndexT{});
        }
        
        StringVecT param_names() const override
        {
            StringVecT names(_num_params);
            append_param_names(names.begin(),IndexT());
            return names;
        }
        
        double cdf(const VecT &u) const override { return cdf(u.begin(),IndexT()); }
        double pdf(const VecT &u) const override { return pdf(u.begin(),IndexT()); }
        double llh(const VecT &u) const override { return llh(u.begin(),IndexT()); }
        double rllh(const VecT &u) const override { return rllh(u.begin(),IndexT()); }
        void grad_accumulate(const VecT &u, VecT &g) const override { grad_accumulate(u,g,IndexT()); }
        void grad2_accumulate(const VecT &u, VecT &g2) const override { grad2_accumulate(u,g2,IndexT()); }
        void hess_accumulate(const VecT &u, MatT &h) const override { hess_accumulate(u,h,IndexT()); }
        void grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2) const override { grad_grad2_accumulate(u,g,g2,IndexT()); }
        void grad_hess_accumulate(const VecT &u, VecT &g, MatT &h) const override { grad_hess_accumulate(u,g,h,IndexT()); }
        
        VecT sample(AnyRngT &rng) override
        {
            VecT s(_num_dim);
            sample(rng, s.begin(), IndexT());
            return s;
        }
        
        MatT sample(AnyRngT &rng, IdxT nSamples) override
        {
            MatT s(_num_dim,nSamples);
            sample(rng, s.begin(), nSamples, IndexT());
            return s;
        }
        
        VecT llh_components(const VecT &theta) const override { return llh_components(theta.begin(), IndexT());}
        VecT rllh_components(const VecT &theta) const override { return rllh_components(theta.begin(), IndexT());}

    private:
        /* Data members */
        std::tuple<Ts...> dists;
        
        template<std::size_t... I> 
        bool is_equal(const DistTuple<Ts...> &o, std::index_sequence<I...>) const
        { return meta::logical_and_in_order( {std::get<I>(dists) == std::get<I>(o.dists)...}); }

        template<class IterT, std::size_t... I> 
        void append_lbound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_lbound(p),0)...} ); }
        
        template<class IterT, std::size_t... I> 
        void append_ubound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_ubound(p),0)...} ); }
        
        template<class IterT, std::size_t... I> 
        void append_global_lbound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_global_lbound(p),0)...} ); }
        
        template<class IterT, std::size_t... I> 
        void append_global_ubound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_global_ubound(p),0)...} ); }
        
        template<class IterT, std::size_t... I> 
        void set_lbound(IterT b, std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_lbound_from_iter(b),0)...} ); }

        template<class IterT, std::size_t... I> 
        void set_ubound(IterT b, std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_ubound_from_iter(b),0)...} ); }

        template<class IterT, std::size_t... I> 
        void set_bounds(IterT lb, IterT ub, std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_bounds_from_iter(lb,ub),0)...} ); }

        template<class IterT, std::size_t... I> 
        void append_params(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_params(p),0)...} ); }

        template<class IterT, std::size_t... I> 
        void set_params(IterT p,std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_params_iter(p),0)...} ); }
 
        template<class IterT, std::size_t... I> 
        bool check_params(IterT p,std::index_sequence<I...>) const
        { return meta::logical_and_in_order( {std::get<I>(dists).check_params_iter(p)...} ); }

        template<class IterT, std::size_t... I> 
        void append_params_lbound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_params_lbound(p),0)...} ); }

        template<class IterT, std::size_t... I> 
        void append_params_ubound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_params_ubound(p),0)...} ); }
 
        template<std::size_t... I> 
        std::vector<VecT> params_components(std::index_sequence<I...>) const
        { return {std::get<I>(dists).params()...}; }

        template<class IterT, std::size_t... I> 
        void append_param_names(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_param_names(p),0)...} ); }

        template<class IterT, std::size_t... I> 
        void set_param_names(IterT p,std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_param_names_iter(p),0)...} ); }
                
        template<class IterT, std::size_t... I> 
        double cdf(IterT u,std::index_sequence<I...>) const
        { return meta::prod_in_order( {std::get<I>(dists).cdf_from_iter(u)...} ); }
        
        template<class IterT, std::size_t... I> 
        double pdf(IterT u,std::index_sequence<I...>) const
        { return meta::prod_in_order( {std::get<I>(dists).pdf_from_iter(u)...} ); }
        
        template<class IterT, std::size_t... I> 
        double llh(IterT u,std::index_sequence<I...>) const
        { return meta::sum_in_order( {std::get<I>(dists).llh_from_iter(u)...} ); }
        
        template<class IterT, std::size_t... I> 
        double rllh(IterT u,std::index_sequence<I...>) const
        { return meta::sum_in_order( {std::get<I>(dists).rllh_from_iter(u)...} ); }
                        
        template<std::size_t... I> 
        void grad_accumulate(const VecT &u, VecT &g,std::index_sequence<I...>) const
        { 
            IdxT k=0;
            meta::call_in_order( {(std::get<I>(dists).grad_accumulate_idx(u,g,k),0)...} ); 
        }
        
        template<std::size_t... I> 
        void grad2_accumulate(const VecT &u, VecT &g2,std::index_sequence<I...>) const 
        { 
            IdxT k=0;
            meta::call_in_order( {(std::get<I>(dists).grad2_accumulate_idx(u,g2,k),0)...} ); 
        }
        
        template<std::size_t... I> 
        void hess_accumulate(const VecT &u, MatT &m, std::index_sequence<I...>) const 
        {
            IdxT k=0;
            meta::call_in_order( {(std::get<I>(dists).hess_accumulate_idx(u,m,k),0)...} );
        }
        
        template<std::size_t... I> 
        void grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2, std::index_sequence<I...>) const 
        { 
            IdxT k=0;
            meta::call_in_order( {(std::get<I>(dists).grad_grad2_accumulate_idx(u,g,g2,k),0)...} );
        }
        
        template<std::size_t... I> 
        void grad_hess_accumulate(const VecT &u, VecT &g, MatT &h,std::index_sequence<I...>) const
        {
            IdxT k=0;
            meta::call_in_order( {(std::get<I>(dists).grad_hess_accumulate_idx(u,g,h,k),0)...} );
        }

        template<class IterT, std::size_t... I> 
        void sample(AnyRngT &rng, IterT s, std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).append_sample(rng,s),0)...} ); }

        template<class IterT, std::size_t... I> 
        void sample(AnyRngT &rng, IterT s, IdxT nSamples, std::index_sequence<I...>)
        {     
            for(IdxT n=0; n<nSamples; n++) 
                meta::call_in_order( {(std::get<I>(dists).append_sample(rng,s),0)...} );
        }
        
        template<class IterT, std::size_t... I> 
        VecT llh_components(IterT theta, std::index_sequence<I...>) const
        { return {std::get<I>(dists).llh_from_iter(theta)...}; }
        
        template<class IterT, std::size_t... I> 
        VecT rllh_components(IterT theta, std::index_sequence<I...>) const
        { return {std::get<I>(dists).rllh_from_iter(theta)...}; }
    }; /* class DistTuple */
    
    class EmptyDistTuple : public DistTupleHandle
    {
    public:  
        EmptyDistTuple() {}
        explicit EmptyDistTuple(const std::tuple<>&) {}
        explicit EmptyDistTuple(std::tuple<>&&) {}
        
        const std::type_info& type_info() const override {return typeid(std::tuple<>);}
        std::unique_ptr<DistTupleHandle> clone() const override {return std::make_unique<EmptyDistTuple>();}
        bool is_equal(const DistTupleHandle &o) const override { return o.num_dists()==0; }
        
        IdxT num_dists() const override {return 0;}
        IdxT num_dim() const override {return 0;}
        IdxT num_params() const override {return 0;}
        UVecT num_dim_components() const override {return {};}
        UVecT num_params_components() const override {return {};}
        TypeInfoVecT component_types() const override {return {};}

        VecT lbound() const override {return {};}
        VecT ubound() const override {return {};}
        VecT global_lbound() const override {return {};}
        VecT global_ubound() const override {return {};}
        void set_lbound(const VecT &new_bound) override 
            {if( !new_bound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        void set_ubound(const VecT &new_bound) override 
            {if( !new_bound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        void set_bounds(const VecT &new_lbound,const VecT &new_ubound) override 
            {if( !new_lbound.is_empty() || !new_ubound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}

        VecT params() const override {return {};}
        void set_params(const VecT &params) override
            {if(!params.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        bool check_params(const VecT &) const override { return true; }        
        VecT params_lbound() const override { return {}; }
        VecT params_ubound() const override { return {}; }
        std::vector<VecT> params_components() const override { return {}; }
        StringVecT param_names() const override {return {};}
        double cdf(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double pdf(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double llh(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double rllh(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }

        void grad_accumulate(const VecT&, VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad2_accumulate(const VecT&, VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void hess_accumulate(const VecT&, MatT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad_grad2_accumulate(const VecT&, VecT&, VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad_hess_accumulate(const VecT&, VecT&, MatT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        VecT sample(AnyRngT&) override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        MatT sample(AnyRngT&, IdxT) override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }

        VecT llh_components(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        VecT rllh_components(const VecT&) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
    }; /* class EmptyDistTuple */
public:
    /* Adaptor for UnivariateDists */
    template<class Dist>
    class ComponentDistAdaptor<Dist,meta::EnableIfSubclassT<Dist,UnivariateDist>> : public Dist {
    public:
        ComponentDistAdaptor() : ComponentDistAdaptor(Dist{}) { }
        explicit ComponentDistAdaptor(Dist &&dist) : Dist(std::move(dist)) { }
        explicit ComponentDistAdaptor(const Dist &dist) : Dist(dist) { }
               
        template<class IterT> void append_lbound(IterT &v) const { *v++ = this->lbound(); } 
        template<class IterT> void append_ubound(IterT &v) const { *v++ = this->ubound(); } 
        template<class IterT> void append_global_lbound(IterT &v) const { *v++ = this->lbound(); } 
        template<class IterT> void append_global_ubound(IterT &v) const { *v++ = this->ubound(); } 
        template<class IterT> void set_lbound_from_iter(IterT& lbounds) { this->set_lbound(*lbounds++); }
        template<class IterT> void set_ubound_from_iter(IterT& ubounds) { this->set_ubound(*ubounds++); }   
        
        template<class IterT> void set_bounds_from_iter(IterT& lbounds, IterT &ubounds) 
        { this->set_bounds(*lbounds++, *ubounds++); }
        
        template<class IterT> void append_params(IterT& v) const 
        { v = std::copy_n(this->params().begin(), Dist::num_params(), v); }
        
        template<class IterT> void append_params_lbound(IterT& v) const 
        { v = std::copy_n(this->param_lbound().begin(), Dist::num_params(), v); }
        
        template<class IterT> void append_params_ubound(IterT& v) const 
        { v = std::copy_n(this->param_ubound().begin(), Dist::num_params(), v); }

        template<class IterT> void append_param_names(IterT& v) const
        { v = std::copy_n(this->param_names().begin(), Dist::num_params(), v); }
                
        template<class IterT> double cdf_from_iter(IterT &u) const { return this->cdf(*u++); }
        template<class IterT> double pdf_from_iter(IterT &u) const { return this->pdf(*u++); }
        template<class IterT> double llh_from_iter(IterT &u) const { return this->llh(*u++); }
        template<class IterT> double rllh_from_iter(IterT &u) const { return this->rllh(*u++); }

        void grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const 
        { 
            g(k) += this->grad(u(k));
            k++;
        }
        
        void grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const 
        {
            g2(k) += this->grad2(u(k));
            k++;
        }
        
        void hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const 
        { 
            h(k,k) += this->grad2(u(k));
            k++;
        }
        
        void grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const 
        { 
            this->grad_grad2_accumulate(u(k),g(k),g2(k));
            k++;
        }
        
        void grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const 
        { 
            this->grad_grad2_accumulate(u(k),g(k),h(k,k)); //hess and grad2 are the same in 1D
            k++;
        }

        template<class RngT, class IterT> 
        void append_sample(RngT &rng, IterT &iter) { *iter++ = this->sample(rng); }
    };

     /* Adaptor for MultivariateDists */
    template<class Dist>
    class ComponentDistAdaptor<Dist,meta::EnableIfSubclassOfNumericTemplateT<Dist,MultivariateDist>> : public Dist {
    public:
        ComponentDistAdaptor() : ComponentDistAdaptor(Dist{}) { }      
        explicit ComponentDistAdaptor(Dist &&dist) : Dist(std::move(dist)) { }
        explicit ComponentDistAdaptor(const Dist &dist) : Dist(dist) { }
                    
        template<class IterT> void append_lbound(IterT &v) const 
        {  v = std::copy_n(this->lbound().begin(),Dist::num_dim(),v); }
        
        template<class IterT> void append_ubound(IterT &v) const 
        {  v = std::copy_n(this->ubound().begin(),Dist::num_dim(),v); }

        template<class IterT> void append_global_lbound(IterT &v) const 
        {  v = std::copy_n(this->global_lbound().begin(),Dist::num_dim(),v); }
        
        template<class IterT> void append_global_ubound(IterT &v) const 
        {  v = std::copy_n(this->global_ubound().begin(),Dist::num_dim(),v); }

        template<class IterT> void set_lbound_from_iter(IterT& v) 
        { 
            typename Dist::NdimVecT b;
            v = std::copy_n(v,Dist::num_dim(),b.begin());
            this->set_lbound(b); 
        }
        
        template<class IterT> void set_ubound_from_iter(IterT& v) 
        { 
            typename Dist::NdimVecT b;
            v = std::copy_n(v,Dist::num_dim(),b.begin());
            this->set_ubound(b); 
        }
        
        template<class IterT> void set_bounds_from_iter(IterT& lb_iter, IterT &ub_iter) 
        { 
            typename Dist::NdimVecT lb,ub;
            lb_iter = std::copy_n(lb_iter,Dist::num_dim(),lb.begin());
            ub_iter = std::copy_n(ub_iter,Dist::num_dim(),ub.begin());
            this->set_bounds(lb,ub); 
        }
        
        template<class IterT> void append_params(IterT& v) const 
        { v = std::copy_n(this->params().begin(), Dist::num_params(), v); }
        

        template<class IterT> void append_params_lbound(IterT& v) const 
        { v = std::copy_n(this->param_lbound().begin(), Dist::num_params(), v); }
        
        template<class IterT> void append_params_ubound(IterT& v) const 
        { v = std::copy_n(this->param_ubound().begin(), Dist::num_params(), v); }

        template<class IterT> void append_param_names(IterT& v) const
        { v = std::copy_n(this->param_names().begin(), Dist::num_params(), v); }
        
        template<class IterT> double cdf_from_iter(IterT &u) const { return this->cdf(*u++); }
        template<class IterT> double pdf_from_iter(IterT &u) const { return this->pdf(*u++); }
        template<class IterT> double llh_from_iter(IterT &u) const { return this->llh(*u++); }
        template<class IterT> double rllh_from_iter(IterT &u) const { return this->rllh(*u++); }

        void grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const 
        { 
            IdxT N = Dist::num_dim();
            g.subvec(k,k+N-1) += this->grad(u.subvec(k,k+N-1));
            k+=N;
        }
        
        void grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const 
        { 
            IdxT N = Dist::num_dim();
            g2.subvec(k,k+N-1) += this->grad2(u.subvec(k,k+N-1));
            k+=N;
        }
        
        void hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const 
        { 
            IdxT N = Dist::num_dim();
            auto H=this->hess(u.subvec(k,k+N-1));
            for(IdxT j=k; j<k+N; j++) for(IdxT i=k; i<k+N; i++) h(k+i,k+j) = H(i,j);
            k+=N;
        }

        
        void grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const 
        { 
            IdxT N = Dist::num_dim();
            auto U = u.subvec(k,k+N-1);
            g.subvec(k,k+N-1) += this->grad(U);
            g2.subvec(k,k+N-1) += this->grad2(U);
            k+=N;
        }
        
        void grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const 
        { 
            IdxT N = Dist::num_dim();
            auto U = u.subvec(k,k+N-1);
            g.subvec(k,k+N-1) += this->grad(U);
            auto H = this->hess(U);
            for(IdxT j=k; j<k+N; j++) for(IdxT i=k; i<k+N; i++) h(k+i,k+j) = H(i,j);
            k+=N;
        }

        template<class RngT, class IterT> 
        void append_sample(RngT &rng, IterT &v) 
        { v = std::copy_n(this->sample(rng).begin(), Dist::num_params(), v); }
    };

    
private:    

    void initialize_from_handle(); //Called on every new handle initialization
    
    static std::string generate_var_name() 
    {
        static IdxT count = 0;
        return std::string("v") + std::to_string(count++);
    }

public:
    template<class DistT> using ComponentDistT = 
                ComponentDistAdaptor<BoundsAdaptedDistT<std::remove_reference_t<DistT>>>;

    /* make_component_dist() */
    //Null-op version if DistT is already wrapped in a ComponentDistAdaptor
    template<class DistT>
    static
    meta::ReturnIfInstantiatedFromT<DistT,DistT,ComponentDistAdaptor>
    make_component_dist(DistT&& dist)
    { return std::forward<DistT>(dist); }

    template<class DistT>
    static
    meta::ReturnIfNotInstantiatedFromT<ComponentDistT<DistT>,DistT,ComponentDistAdaptor>
    make_component_dist(DistT&& dist)
    { return ComponentDistT<DistT>{make_adapted_bounded_dist(std::forward<DistT>(dist))}; }

//     //Univariate wrapper
//     template<class DistT>
//     meta::ReturnIfNotInstantiatedFromT<ComponentDistT<DistT>,
//             meta::ReturnIfSubclassT<DistT, DistT, UnivariateDist>,ComponentDistAdaptor>
//     make_component_dist(DistT&& dist)
//     { return ComponentDistT<DistT>{make_adapted_bounded_dist(std::forward<DistT>(dist))}; }
//     
//     //Mutlivariate wrapper
//     template<class DistT>
//     meta::ReturnIfNotInstantiatedFromT<ComponentDistT<DistT>,
//             meta::ReturnIfSubclassOfNumericTemplateT<DistT, DistT, MultivariateDist>,ComponentDistAdaptor>
//     make_component_dist(DistT&& dist)
//     { return ComponentDistT<DistT>{make_adapted_bounded_dist(std::forward<DistT>(dist))}; }
 
    /* make_component_dist_tuple()
     * Uses perfect forwarding.  1-form for (const&) one for (&&).  Each method has a helper template with extra integer parameters
     * to enable iterating the tuple.
     */
    template<class... Ts>
    static
    std::tuple<ComponentDistT<Ts>...>
    make_component_dist_tuple(const std::tuple<Ts...>& dists)
    { return make_component_dist_tuple(dists,std::index_sequence_for<Ts...>{}); }
    
    //helper
    template<class... Ts,std::size_t... I>
    static
    std::tuple<ComponentDistT<Ts>...>
    make_component_dist_tuple(const std::tuple<Ts...>& dists, std::index_sequence<I...> )
    { return std::make_tuple(make_component_dist(make_adapted_bounded_dist(std::get<I>(dists)))...); }

    template<class... Ts>
    static
    std::tuple<ComponentDistT<Ts>...>
    make_component_dist_tuple(std::tuple<Ts...>&& dists)
    { return make_component_dist_tuple(std::move(dists),std::index_sequence_for<Ts...>{}); }
    
    //helper
    template<class... Ts,std::size_t... I>
    static
    std::tuple<ComponentDistT<Ts>...>
    make_component_dist_tuple(std::tuple<Ts...>&& dists, std::index_sequence<I...> )
    { return std::make_tuple(make_component_dist(make_adapted_bounded_dist(std::get<I>(dists)))...); }    

private:     
    /* Private Memeber variables */
    std::unique_ptr<DistTupleHandle> handle;

    /* Param name index */
    using ParamNameMapT = std::unordered_map<std::string,int>;    
    static ParamNameMapT initialize_param_name_idx(const StringVecT &names);// throw (ParameterNameUniquenessError)
    mutable ParamNameMapT param_name_idx;
    
    mutable StringVecT _component_names;
    mutable StringVecT _dim_variables;
    mutable StringVecT _param_names;
    mutable bool component_names_initialized;
    mutable bool dim_variables_initialized;
    mutable bool param_names_initialized;
    
    void initialize_component_names() const;
    void initialize_dim_variables() const;
    void initialize_param_names() const;

};

/* CompositeDist<RngT> template methods */

// template<class... Ts, typename=meta::EnableIfAllAreNotTupleT<Ts...>>
// void CompositeDist::initialize(Ts&&... dists)
// {
//     handle = std::unique_ptr<DistTupleHandle>{ 
//                 new DistTuple<ComponentDistT<Ts>...>{make_component_dist(std::forward<Ts>(dists))...} };
//     initialize_from_handle();
// }
// 
// template<class... Ts, typename=meta::EnableIfAllAreNotTupleT<Ts...>>
// void CompositeDist::initialize(std::tuple<Ts...>&& dist_tuple)
// {
//     handle = std::unique_ptr<DistTupleHandle>{ 
//                 new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(std::move(dist_tuple))} };
//     initialize_from_handle();
// }
// 
// 
// template<class... Ts, typename=meta::EnableIfNonEmpty<Ts...>>
// void CompositeDist::initialize(const std::tuple<Ts...>& dist_tuple)
// {
//     handle = std::unique_ptr<DistTupleHandle>{ 
//                 new DistTuple<ComponentDistT<Ts>...>{make_component_dist_tuple(dist_tuple)} };
//     initialize_from_handle();
// }

template<class... Ts> 
const std::tuple<Ts...>& 
CompositeDist::get_dist_tuple() const 
{
    const std::type_info& tuple_id = typeid(std::tuple<Ts...>);
    if (typeid(std::tuple<Ts...>) != handle->type_info()){
        std::ostringstream os;
        os<<"CompositeDist Expected type_id:" << handle->type_info().name() << " got type_id:"<<tuple_id.name();
        throw RuntimeTypeError(os.str());
    } else {
        return static_cast<std::unique_ptr<DistTuple<Ts...>>>(handle)->dists;
    }
}

template<class StringVec>
void CompositeDist::set_component_names(StringVec &&names)
{
    if(names.size() != num_components()) {
        std::ostringstream msg;
        msg<<"Expected: "<<num_components()<<" names. Got: "<<names.size();
        throw ParameterSizeError(msg.str());
    }
    _component_names = std::forward<StringVec>(names);
    component_names_initialized = true;
}

template<class StringVec>
void CompositeDist::set_dim_variables(StringVec &&names)
{
    if(names.size() != num_dim()) {
        std::ostringstream msg;
        msg<<"Expected: "<<num_dim()<<" names. Got: "<<names.size();
        throw ParameterSizeError(msg.str());
    }
    _dim_variables = std::forward<StringVec>(names);
    dim_variables_initialized = true;
}

template<class StringVec>
void CompositeDist::set_param_names(StringVec &&names)
{
    if(names.size() != num_params()) {
        std::ostringstream msg;
        msg<<"Expected: "<<num_params()<<" names. Got: "<<names.size();
        throw ParameterSizeError(msg.str());
    }
    _param_names = std::forward<StringVec>(names);
    param_name_idx = initialize_param_name_idx(param_names());
    param_names_initialized = true;
}

/* Protected methods */


} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_COMPOSITEDIST_H */
