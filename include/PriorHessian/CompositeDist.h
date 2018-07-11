/** @file CompositeDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief The Frank copula computations
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_COMPOSITEDIST_H
#define _PRIOR_HESSIAN_COMPOSITEDIST_H

#include<utility>
#include<memory>
#include<unordered_map>
#include<sstream>

#include<armadillo>

#include "PriorHessian/Meta.h"
#include "PriorHessian/util.h"
#include "PriorHessian/PriorHessianError.h"

#include "ParallelRngManager/AnyRng/AnyRng.h"

namespace prior_hessian {

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
    using AnyRngT = any_rng::AnyRngT;
    
    CompositeDist();
    
    /** @brief Initialize from a tuple of of UnivariateDist's or MulitvariateDist's */
    template<class... Ts,typename=meta::EnableConstructorIfAllAreNotTupleAndAreNotSelfT<CompositeDist,Ts...>>
    explicit CompositeDist(Ts&&... dists);
    template<class... Ts>
    explicit CompositeDist(std::tuple<Ts...>&& dist_tuple);
    template<class... Ts>
    explicit CompositeDist(const std::tuple<Ts...>& dist_tuple);
    
    void initialize();
    template<class... Ts, typename=meta::EnableIfAllAreNotTupleT<Ts...>>
    void initialize(Ts&&... dists);
    template<class... Ts>
    void initialize(std::tuple<Ts...>&& dist_tuple);
    template<class... Ts>
    void initialize(const std::tuple<Ts...>& dist_tuple);
    
    CompositeDist(const CompositeDist &)
    CompositeDist& operator=(const CompositeDist &);     
    CompositeDist(CompositeDist&&);
    CompositeDist& operator=(CompositeDist&&);  
    
    void clear()
    {
        handle = std::unique_ptr<DistTupleHandle>(new EmptyDistTuple());
        initialize_from_handle();
    }
       
    template<class... Ts> 
    const std::tuple<Ts...>& get_dist_tuple() const; 
    
    IdxT num_component_dists() const { return handle->num_dists(); }
    TypeInfoVecT component_types() const { return handle->component_types(); }
    
    /* Dimensionality and variable names */
    IdxT num_dim() const { return handle->num_dim(); }
    UVecT components_num_dim() const { return handle->components_num_dim(); }
    StringVecT dim_variables() const { return handle->dim_variables(); }
    void set_dim_variables(const StringVecT &vars)  { handle->set_dim_variables(vars); }

    
    /* Bounds */
    const VecT& lbound() const { return handle->lbound(); }
    const VecT& ubound() const { return handle->ubound(); }
    bool in_bounds(const VecT &u) const { return arma::all(_lbound<u && u<_ubound); }
    void set_lbound(const VecT &hew_bound) { handle->set_lbound(new_bound); }
    void set_ubound(const VecT &hew_bound) { handle->set_ubound(new_bound); }
    void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) { handle->set_bounds(new_lbound, new_ubound); }

    /* Distribution Parameters */
    IdxT num_params() const { return handle->num_params(); }
    UVecT components_num_params() const { return handle->components_num_params(); }
    VecT params() const { return handle->params(); } 
    void set_params(const VecT &params) { handle->set_params(new_params); }

    /* Distribution Parameter Descriptions (names) */
    StringVecT param_names() const { return handle->param_names(); }

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
        VecT g(_num_dim, arma::fill::zeros);
        handle->grad_accumulate(u,g);
        return g;
    }

    VecT grad2(const VecT &u) const
    {
        VecT g2(_num_dim, arma::fill::zeros);
        handle->grad2_accumulate(u,g2);
        return g2;
    }

    MatT hess(const VecT &u) const
    {
        MatT h(_num_dim,_num_dim,arma::fill::zeros);
        handle->hess_accumulate(u,h);
        return h;
    }
    
    /* Accumulate methods add thier contributions to respective grad, grad2 or hess arguments 
     * These are more efficient.  They use the Dist.xxx_acumulate methods, which are intern provided by the
     * CRTP UnivariateDist and MulitvariateDists base class templates.
     * 
     * All this technology should make these calls very close to custom coded grad and hess calls for any combination of dists.
     */
    void grad_accumulate(const VecT &u, VecT &grad) const { return handle->grad_accumulate(theta,grad); }
    void grad2_accumulate(const VecT &u, VecT &grad2) const { return handle->grad2_accumulate(theta,grad2); }
    void hess_accumulate(const VecT &u, MatT &hess) const { return handle->hess_accumulate(theta,hess); }
    void grad_grad2_accumulate(const VecT &u, VecT &grad, VecT &grad2) const { return handle->grad_grad2_accumulate(theta,grad,grad2); }
    void grad_hess_accumulate(const VecT &u, VecT &grad, MatT &hess) const { return handle->grad_hess_accumulate(theta,grad,hess); }
    //Convenience methods for the lazy.
    VecT make_zero_grad() const { return {num_dim(),arma::fill::zeros}; }
    MatT make_zero_hess() const { return {num_dim(),num_dim(),arma::fill::zeros}; }
    
    VecT sample(AnyRngT &rng) { return handle->sample(rng); }
    MatT sample(AnyRngT &rng, IdxT num_samples) { return handle->sample(rng,num_samples); }

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
        virtual IdxT num_dists() const = 0;
        virtual TypeInfoVecT component_types() const = 0;
        virtual IdxT num_dim() const = 0;
        virtual UVecT components_num_dim() const = 0;
        virtual StringVecT dim_variables() const = 0;
        virtual void set_dim_variables(const StringVecT &vars) = 0;
        virtual VecT lbound() const = 0;
        virtual VecT ubound() const = 0;
        virtual void set_lbound(const VecT &hew_lbound) = 0;
        virtual void set_ubound(const VecT &hew_ubound) = 0;
        virtual void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) = 0;
        virtual IdxT num_params() const = 0;
        virtual UVecT components_num_params() const = 0;
        virtual VecT params() const = 0;        
        virtual void set_params(const VecT &params) = 0;
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
        constexpr static IdxT _num_dim = meta::unordered_sum(Ts::num_dim()...);
        constexpr static IdxT _num_params = meta::unordered_sum(Ts::num_params()...);
        using StaticSizeArrayT = std::array<IdxT,_num_dists>;
        constexpr static StaticSizeArrayT _component_num_dim = {{Ts::num_dim()...}}; 
        constexpr static StaticSizeArrayT _component_num_params = {{Ts::num_params()...}}; 
    public:  
        explicit DistTuple(const std::tuple<Ts...> &dists) : dists(std::move(_dists)) { }
        explicit DistTuple(std::tuple<Ts...>&& dists) : dists(_dists) { }
        
        const std::type_info& type_info() const override { return typeid(dists); }
        std::unique_ptr<DistTupleHandle> clone() const override { return std::make_unique<DistTuple<Ts...>>(dists); }
        constexpr IdxT num_dists() const override { return _num_dists; }
        constexpr IdxT num_dim() const override { return _num_dim; }
        constexpr IdxT num_params() const override { return _num_params; }
        UVecT components_num_dim() const override { return {Ts::num_dim()...}; }
        UVecT components_num_params() const override { return {Ts::num_params()...}; }
        TypeInfoVecT component_types() const override { return {std::type_index(typeid(Ts))...}; }

        StringVecT dim_variables() const override
        {
            StringVecT vars(_num_dim);
            append_dim_variables(vars.begin(),IndexT());
            return vars;
        }

        void set_dim_variables(const StringVecT &vars) override { set_dim_variables(vars.begin(),IndexT()); }
        
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
        
        void set_lbound(const VecT &hew_bound) override { set_lbound(new_bound.begin(), IndexT{}); }
        void set_ubound(const VecT &hew_bound) override { set_ubound(new_bound.begin(), IndexT{}); }
        void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) override { set_bounds(new_lbound.begin(), new_ubound.begin(), IndexT{}); }

        VecT params() const override
        {
            VecT params(_num_params);
            append_params(params.begin(),IndexT());
            return params;
        }
        
        void set_params(const VecT &params) override { set_params(params.begin(),IndexT()); }
        
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
        
        VecT sample(RngT &rng) override
        {
            VecT s(_num_dim);
            sample(rng, s.begin(), IndexT());
            return s;
        }
        
        MatT sample(RngT &rng, IdxT nSamples) override
        {
            MatT s(_num_dim,nSamples);
            sample(rng, s.begin(), nSamples, IndexT());
            return s;
        }
        
        VecT llh_components(const VecT &u) const override { return llh_components(theta.begin(), IndexT());}
        VecT rllh_components(const VecT &u) const override { return rllh_components(theta.begin(), IndexT());}

    private:
        /* Data members */
        std::tuple<Ts...> dists;
        
        template<class IterT, std::size_t... I> 
        void append_dim_variables(IterT v, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_var_name(v),0)...} ); }

        template<class IterT, std::size_t... I> 
        void set_dim_variables(IterT v,std::index_sequence<I...>)
        { meta::call_in_order( {(std::get<I>(dists).set_var_name_iter(v),0)...} ); }  
                
        template<class IterT, std::size_t... I> 
        void append_lbound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_lbound(p),0)...} ); }
        
        template<class IterT, std::size_t... I> 
        void append_ubound(IterT p, std::index_sequence<I...>) const
        { meta::call_in_order( {(std::get<I>(dists).append_ubound(p),0)...} ); }
        
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
        void sample(AnyRngT &rng, IterT s, std::index_sequence<I...>);
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
        
        constexpr IdxT num_dists() const override {return 0;}
        constexpr IdxT num_dim() const override {return 0;}
        constexpr IdxT num_params() const override {return 0;}
        UVecT components_num_dim() const override {return {};}
        UVecT components_num_params() const override {return {};}
        TypeInfoVecT component_types() const override {return {};}

        StringVecT dim_variables() const override {return {};}
        void set_dim_variables(const StringVecT &vars) override {if(!vars.empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        VecT lbound() const override {return {};}
        VecT ubound() const override {return {};}
        void set_lbound(const VecT &new_bound) override 
            {if( !new_bound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        void set_ubound(const VecT &new_bound) override 
            {if( !new_bound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        void set_bounds(const VecT &new_lbound,const VecT &new_ubound) override 
            {if( !new_lbound.is_empty() || !new_ubound.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}

        VecT params() const override {return {};}
        void set_params(const VecT &params) override
            {if(!params.is_empty()) throw RuntimeTypeError("Empty dist tuple cannot be set.");}
        StringVecT param_names() const override {return {};}
        double cdf(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double pdf(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double llh(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        double rllh(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }

        void grad_accumulate(const VecT &u, VecT &g) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad2_accumulate(const VecT &u, VecT &g2) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void hess_accumulate(const VecT &u, MatT &h) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        void grad_hess_accumulate(const VecT &u, VecT &g, MatT &h) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        VecT sample(AnyRngT &rng) override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        MatT sample(AnyRngT &rng, IdxT nSamples) override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }

        VecT llh_components(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
        VecT rllh_components(const VecT &u) const override { throw RuntimeTypeError("Empty dist cannot be evaluated."); }
    }; /* class EmptyDistTuple */


   /** @brief An internal class that wraps conforming dists with adjustable bounds.
     * 
     * 
     */
    template<class Dist>
    class ComponentDistAdaptor : public Dist {   
    public:
        ComponentDistAdaptor() 
            : ComponentDistAdaptor(Dist{}, generate_var_name()) { }
        
        explicit ComponentDistAdaptor(std::string var_name) 
            : ComponentDistAdaptor(Dist{}, std::move(var_name)) { }
        
        explicit ComponentDistAdaptor(Dist &&dist) 
            : ComponentDistAdaptor(std::move(dist), generate_var_name()) { }        
        
        explicit ComponentDistAdaptor(const Dist &dist) 
            : ComponentDistAdaptor(dist, generate_var_name()) { }
        
        ComponentDistAdaptor(Dist &&dist,  StringVecT &&var_name) 
            : ComponentDistAdaptor(std::move(dist), std::move(var_name[0])) { }
        
        ComponentDistAdaptor(const Dist &dist, StringVecT var_name)  
            : ComponentDistAdaptor(dist, std::move(var_name[0])) { }
        
        ComponentDistAdaptor(Dist &&dist, std::string &&var_name)  
            : _dist(std::move(dist)), _var_name{std::move(var_name)} { }
        
        ComponentDistAdaptor(const Dist &dist, std::string var_name)  
            : _dist(dist), _var_name{std::move(var_name)} { }
        
        static constexpr IdxT num_dim() { return Dist::num_dim(); }
        static constexpr IdxT num_params() { return Dist::num_params(); }
        const StringVecT& var_names() const { return {_var_name}; }
        const std::string& var_name(IdxT i) const { return _var_name; }
        void set_var_names(StringVecT var_name) { _var_name = std::move(var_names[0]); };
        void set_var_name(IdxT i, std::string new_var_name) { _var_name = std::move(new_var_name); }
        
        StringVecT param_names() const;
        std::string param_name(IdxT i) const;
            
        Dist& get_dist() { return _dist; }
        const Dist& get_dist() const { return _dist; }
        void set_dist(Dist&& dist) { _dist = std::move(dist); }
        void set_dist(const Dist& dist) { _dist = dist; }
        void set_dist(Dist&& dist, std::string var_name) 
        { 
            _dist = std::move(dist); 
            _var_name = std::move(var_name);
        }
        
        void set_dist(const Dist& dist, std::string var_name) 
        { 
            _dist = dist; 
            _var_name = std::move(var_name);
        }

    private:
        Dist _dist;
        std::string _var_name;

        static std::string generate_var_name() 
        {
            static IdxT count = 0;
            return "v"s + std::to_string(count++);
        }
        
        template<class IterT> void append_var_name(IterT &v) const { *v++ = _var_name; }
        template<class IterT> void set_var_name_iter(IterT &v) { _var_name = *v++; } 
        template<class IterT> void append_lbound(IterT &v) const { *v++ = this->lbound(); } 
        template<class IterT> void append_ubound(IterT &v) const { *v++ = this->ubound(); } 
        template<class IterT> void set_lbound_from_iter(IterT& lbounds) { this->set_lbounds(*lbounds++); }
        template<class IterT> void set_ubound_from_iter(IterT& ubounds) { this->set_ubounds(*ubounds++); }   
        template<class IterT> void set_bounds_from_iter(IterT& lbounds, IterT &ubounds) { this->set_bounds(*lbounds++, *ubounds++); }
        template<class IterT> void append_params(IterT& v) const { for(IdxT n=0;n<this->num_parms();n++) *v++ = this->get_param(n); }
        template<class IterT> void set_params_iter(IterT& v) { for(IdxT n=0;n<this->num_parms();n++) this->set_param(n, *v++); }
        template<class IterT> void append_param_names(IterT& v) const { v = std::copy(this->param_names.cbegin(), this->param_names.cend(), v); }
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

private:    
    void initialize_from_handle();

    template<class... Ts>
    void _initialize_from_components(Ts&&... dists);
    
    std::unique_ptr<DistTupleHandle> handle;

    using ParamNameMapT = std::unordered_map<std::string,int>;    
    static ParamNameMapT initialize_param_name_idx(const StringVecT &names);// throw (ParameterNameUniquenessError)
    ParamNameMapT param_name_idx;
};

/* Constructor free functions and adaptor functions */

namespace detail
{
    template<class Dist>
    class dist_adaptor_traits {
    public:
        using bounds_adapted_dist = void;
        static constexpr bool adaptable_bounds = false;
    };
    
    class NormalDist;
    template<> class dist_adaptor_traits<NormalDist>;
    class GammalDist;
    template<> class dist_adaptor_traits<GammaDist>;
    class ParetolDist;
    template<> class dist_adaptor_traits<ParetoDist>;
    class SymmetricBetaDist;
    template<> class dist_adaptor_traits<SymmetricBetaDist>;
} /* namespace detail */

/** Type traits class for distribution type DistT.  
 *
 * The traits class describes the Adaptor classes applicable to each individual distribution
 */
template<class DistT> using DistTraitsT = detail::dist_adaptor_traits<std::decay_t<DistT>>;

/** The bounds-adapted distribution type for a given distribution type DistT
 * This is the adapted version of the class, i.e., the class that allows truncation or scaling so that the lower and upper bounds are settable.
 */
template<class DistT> using BoundsAdaptedDistT = typename detail::dist_adaptor_traits<std::decay_t<DistT>>::bounds_adapted_dist;


namespace detail
{
    template<class... Ts,std::size_t... I>
    std::tuple<BoundsAdaptedDistT<Ts>...>
    make_adapted_bounded_dist_tuple(std::tuple<Ts...>&& dists, std::index_sequence<I...> )
    {
        return std::make_tuple(make_adapted_bounded_dist(std::get<I>(dists))...);
    }

    template<class... Ts,std::size_t... I>
    std::tuple<BoundsAdaptedDistT<Ts>...>
    make_adapted_bounded_dist_tuple(const std::tuple<Ts...>& dists, std::index_sequence<I...> )
    {
        return std::make_tuple(make_adapted_bounded_dist(std::get<I>(dists))...);
    }

    template<class... Ts,std::size_t... I>
    std::tuple<ComponentDistAdaptor<BoundsAdaptedDistT<Ts>>...>
    make_component_dist_tuple(const std::tuple<Ts...>& dists, std::index_sequence<I...> )
    {
        return std::make_tuple(ComponentDistAdaptor<BoundsAdaptedDistT<Ts>>{make_adapted_bounded_dist(std::get<I>(dists))}...);
    }
} /* namespace detail */


template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
ComponentDistAdaptor<BoundsAdaptedDistT<Dist>>
make_component_dist(Dist &&dist)
{
    return ComponentDistAdaptor<BoundsAdaptedDistT<Dist>>(make_adapted_bounded_dist(std::forward<Dist>(dist)));
}

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
meta::ReturnIfT<Dist, DistTraitsT<Dist>::adaptable_bounds>
make_adapted_bounded_dist(Dist &&dist)
{ return dist; }

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
meta::ReturnIfT<BoundsAdaptedDistT<Dist>, DistTraitsT<Dist>::adaptable_bounds>
make_adapted_bounded_dist(Dist &&dist)
{ return {std::forward<Dist>(dist)}; }

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
meta::ReturnIfT<Dist, DistTraitsT<Dist>::adaptable_bounds>
make_adapted_bounded_dist(Dist &&dist, double lbound, double ubound)
{ 
    dist.set_bounds(lbound,ubound);
    return dist; 
}

template<class Dist, typename=meta::EnableIfIsNotTupleT<Dist>>
meta::ReturnIfT<BoundsAdaptedDistT<Dist>, DistTraitsT<Dist>::adaptable_bounds>
make_adapted_bounded_dist(Dist &&dist, double lbound, double ubound)
{ return {std::forward<Dist>(dist),lbound,ubound}; }

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


/* CompositeDist<RngT> template methods */
template<class... Ts>
CompositeDist::CompositeDist(std::tuple<Ts...>&& dist_tuple) 
    : handle{ std::unique_ptr<DistTupleHandle>{
                new DistTuple<Ts...>( make_component_dist_tuple(std::move(dist_tuple)) ) }}
{ initialize_from_handle(); }

template<class... Ts>
CompositeDist::CompositeDist(const std::tuple<Ts...>& dist_tuple) 
    : handle{ std::unique_ptr<DistTupleHandle>{
                new DistTuple<Ts...>( make_component_dist_tuple(dist_tuple) ) }}
{ initialize_from_handle(); }

template<class... Ts> 
const std::tuple<Ts...>& 
CompositeDist::get_dist_tuple() const 
{
    const std::type_info& tuple_id = typeid(std::tuple<Ts...>);
    if (typeid(std::tuple<Ts...>) != handle->type_info()){
        std::ostringstream os;
        os<<"CompositeDist Expected type_id:"<<handle->type_info()<<" got type_id:"<<tuple_id;
        throw RuntimeTypeError(os.str());
    } else {
        return static_cast<std::unique_ptr<DistTuple<Ts...>>>(handle)->dists;
    }
}

template<class... Ts>
void CompositeDist::initialize(Ts&&... dists)
{
    _initialize_from_components( make_component_dist(std::forward<Ts>(dists)) ... );
}    

template<class... Ts>
void CompositeDist::_initialize_from_components(Ts&&... dists)
{
    handle = std::unique_ptr<DistTupleHandle>{ new DistTuple<Ts...>(std::forward<Ts>(dists)...) };
    initialize_from_handle();
}

template<class... Ts>
void CompositeDist::initialize(std::tuple<Ts...>&& dist_tuple)
{
    handle = std::unique_ptr<DistTupleHandle>{ new DistTuple<Ts...>(make_component_dist_tuple(std::move(dist_tuple))) };
    initialize_from_handle();
}

template<class Dist>
StringVecT CompositeDist::ComponentDistAdaptor<Dist>::param_names() const
{ 
    StringVecT names = static_cast<const Dist*>(this)->param_names;
    std::string prefix = _var_name + '.';
    for(auto &n: names) n.insert(0,prefix);
    return names;
}
    
template<class Dist>
std::string CompositeDist::ComponentDistAdaptor<Dist>::param_name(IdxT i) const
{
    DEBUG_ASSERT(i < num_params(), assert::handler{}, "Parameter index out-of-range");
    std::ostringstream out;
    out<<_var_name<<"."<< static_cast<const Dist*>(this)->param_names[i];
    return out.str();
}


/* Protected methods */


} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_COMPOSITEDIST_H */
