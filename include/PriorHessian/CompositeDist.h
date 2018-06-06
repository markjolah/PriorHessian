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
#include<armadillo>
#include "PriorHessian/BaseDist.h"
#include "PriorHessian/UnivariateDist.h"


namespace prior_hessian {

/** @brief A probability distribution made of independent component distributions composing groups of 1 or more variables.
 * 
 * 
 */
template<class RngT>
class CompositeDist
{
public:
    CompositeDist() = default;
    
    /** @brief Initialize from variable number of UnivariateDist's or MulitvariateDist's arguments */
    template<class... Ts>
    CompositeDist(Ts&&... dists);

    /** @brief Initialize from a tuple of of UnivariateDist's or MulitvariateDist's */
    template<class... Ts>
    CompositeDist(std::tuple<Ts...>&& dist_tuple);
    
    /* Move only type */
    CompositeDist(const CompositeDist &) = delete; 
    CompositeDist& operator=(const CompositeDist &) = delete;     
    CompositeDist(CompositeDist&&) = default;
    CompositeDist& operator=(CompositeDist&&) = default;  
    
    template<class... Ts>
    void initialize(Ts&&... dists);
    template<class... Ts>
    void initialize(std::tuple<Ts...>&& dist_tuple);
    
    IdxT num_component_dists() const; 
    TypeInfoVecT component_types() const;
    
    /* Dimensionality and variable names */
    IdxT num_dim() const;
    UVecT components_num_dim() const;
    StringVecT dim_variables() const;
    void set_dim_variables(const StringVecT &vars);
    
    /* Bounds */
    const VecT& lbound() const;
    const VecT& ubound() const;
    bool in_bounds(const VecT &u) const;
    void set_lbound(const VecT &hew_bound);
    void set_ubound(const VecT &hew_bound);
    void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound);

    /* Distribution Parameters */
    IdxT num_params() const; 
    UVecT components_num_params() const;
    VecT params() const;
    void set_params(const VecT &params);

    /* Distribution Parameter Descriptions (names) */
    StringVecT params_desc() const;
    void set_params_desc(const StringVecT &desc);
    
    //Functions mapped over underlying distributions
    double cdf(const VecT &u) const;
    double pdf(const VecT &u) const;
    double llh(const VecT &u) const;
    double rllh(const VecT &u) const;    
    VecT grad(const VecT &u) const; 
    VecT grad2(const VecT &u) const;
    MatT hess(const VecT &u) const; 
    
    /* Accumulate methods add thier contributions to respective grad, grad2 or hess arguments 
     * These are more efficient.  They use the Dist.xxx_acumulate methods, which are intern provided by the
     * CRTP UnivariateDist and MulitvariateDists base class templates.
     * 
     * All this technology should make these calls very close to custom coded grad and hess calls for any combination of dists.
     */
    void grad_accumulate(const VecT &u, VecT &grad) const;
    void grad2_accumulate(const VecT &u, VecT &grad2) const;
    void hess_accumulate(const VecT &u, MatT &hess) const;
    void grad_grad2_accumulate(const VecT &u, VecT &grad, VecT &grad2) const;
    void grad_hess_accumulate(const VecT &u, VecT &grad, MatT &hess) const;
    //Convenience methods for the lazy.
    VecT make_zero_grad() const { return {num_dim(),arma::fill::zeros}; }
    MatT make_zero_hess() const { return {num_dim(),num_dim(),arma::fill::zeros}; }
    
    VecT sample(RngT &rng);
    MatT sample(RngT &rng, IdxT num_samples);

    /* Per-component values for debugging and plotting purposes */
    VecT llh_components(const VecT &u) const;
    VecT rllh_components(const VecT &u) const;    
protected:
    
    /** @brief Model interface
     */
    class DistTupleHandle
    {
    public:
        virtual ~DistTupleHandle() = default;
        virtual IdxT num_dists() const = 0;
        virtual TypeInfoVecT component_types() const = 0;

        virtual IdxT num_dim() const = 0;
        virtual UVecT components_num_dim() const = 0;
        virtual StringVecT dim_variables() const = 0;
        virtual void set_dim_variables(const StringVecT &vars) = 0;

        virtual VecT lbound() const = 0;
        virtual VecT ubound() const = 0;
        virtual void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) = 0;
    
        virtual IdxT num_params() const = 0;
        virtual UVecT components_num_params() const = 0;
        virtual VecT params() const = 0;
        
        virtual void set_params(const VecT &params) = 0;
        virtual StringVecT params_desc() const = 0;
        virtual void set_params_desc(const StringVecT &desc) = 0;

        virtual double cdf(const VecT &u) const = 0;
        virtual double pdf(const VecT &u) const = 0;
        virtual double llh(const VecT &u) const = 0;
        virtual double rllh(const VecT &u) const = 0;
        
        
        virtual void grad_accumulate(const VecT &u, VecT &grad) const = 0;
        virtual void grad2_accumulate(const VecT &u, VecT &grad2) const = 0;
        virtual void hess_accumulate(const VecT &u, MatT &hess) const = 0;
        virtual void grad_grad2_accumulate(const VecT &u, VecT &grad, VecT &grad2) const = 0;
        virtual void grad_hess_accumulate(const VecT &u, VecT &grad, MatT &hess) const = 0;
        
        virtual VecT sample(RngT &rng) = 0;
        virtual MatT sample(RngT &rng, IdxT nSamples) = 0;

        virtual VecT llh_components(const VecT &u) const = 0;
        virtual VecT rllh_components(const VecT &u) const = 0;
    };
    
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
        DistTuple(std::tuple<Ts...>&& dists);
        DistTuple(DistTuple&&) = default;
        DistTuple(const DistTuple&) = delete;
        
        
        constexpr IdxT num_dists() const override;
        constexpr IdxT num_dim() const override;
        constexpr IdxT num_params() const override;
        UVecT components_num_dim() const override;
        UVecT components_num_params() const override;   
        TypeInfoVecT component_types() const override;        

        StringVecT dim_variables() const override;
        void set_dim_variables(const StringVecT &vars) override;
        VecT lbound() const override;
        VecT ubound() const override;
        void set_bounds(const VecT &hew_lbound,const VecT &hew_ubound) override;

        VecT params() const override;
        void set_params(const VecT &params) override;
        StringVecT params_desc() const override;
        void set_params_desc(const StringVecT &desc) override;
        double cdf(const VecT &u) const override;
        double pdf(const VecT &u) const override;
        double llh(const VecT &u) const override;
        double rllh(const VecT &u) const override;
        

        void grad_accumulate(const VecT &u, VecT &g) const override;
        void grad2_accumulate(const VecT &u, VecT &g2) const override; 
        void hess_accumulate(const VecT &u, MatT &h) const override;
        void grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2) const override;
        void grad_hess_accumulate(const VecT &u, VecT &g, MatT &h) const override;
        VecT sample(RngT &rng) override;
        MatT sample(RngT &rng, IdxT nSamples) override;

        VecT llh_components(const VecT &u) const override;
        VecT rllh_components(const VecT &u) const override;
    private:
        std::tuple<Ts...> dists;
        
        template<class IterT, std::size_t... I> 
        void append_dim_variables(IterT v, std::index_sequence<I...>) const;

        template<class IterT, std::size_t... I> 
        void set_dim_variables(IterT v,std::index_sequence<I...>);
                
        template<class IterT, std::size_t... I> 
        void append_lbound(IterT p, std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        void append_ubound(IterT p, std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        void set_bounds(IterT lb, IterT ub, std::index_sequence<I...>);

        template<class IterT, std::size_t... I> 
        void append_params(IterT p, std::index_sequence<I...>) const;

        template<class IterT, std::size_t... I> 
        void set_params(IterT p,std::index_sequence<I...>);
        
        template<class IterT, std::size_t... I> 
        void append_params_desc(IterT p, std::index_sequence<I...>) const;

        template<class IterT, std::size_t... I> 
        void set_params_desc(IterT p,std::index_sequence<I...>);
                
        template<class IterT, std::size_t... I> 
        double cdf(IterT u,std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        double pdf(IterT u,std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        double llh(IterT u,std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        double rllh(IterT u,std::index_sequence<I...>) const;
                        
        template<std::size_t... I> 
        void grad_accumulate(const VecT &u, VecT &g,std::index_sequence<I...>) const;
        
        template<std::size_t... I> 
        void grad2_accumulate(const VecT &u, VecT &g2,std::index_sequence<I...>) const;

        template<std::size_t... I> 
        void hess_accumulate(const VecT &u, MatT &m, std::index_sequence<I...>) const;
        
        template<std::size_t... I> 
        void grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2, std::index_sequence<I...>) const;
        
        template<std::size_t... I> 
        void grad_hess_accumulate(const VecT &u, VecT &g, MatT &h,std::index_sequence<I...>) const;

        template<class IterT, std::size_t... I> 
        void sample(RngT &rng, IterT s, std::index_sequence<I...>);

        template<class IterT, std::size_t... I> 
        void sample(RngT &rng, IterT s, IdxT nSamples, std::index_sequence<I...>);
        
        template<class IterT, std::size_t... I> 
        VecT llh_components(IterT theta, std::index_sequence<I...>) const;
        
        template<class IterT, std::size_t... I> 
        VecT rllh_components(IterT theta, std::index_sequence<I...>) const;
    };

    /* Protected methods */
    void update_bounds();
    /* Protected member variables */
    std::unique_ptr<DistTupleHandle> handle;
    IdxT _num_dim;
    IdxT _num_params;
    VecT _lbound; 
    VecT _ubound; 
};




/* CompositeDist<RngT> template methods */


template<class RngT>
template<class... Ts>
CompositeDist<RngT>::CompositeDist(Ts&&... dists) : 
    handle( std::unique_ptr<DistTupleHandle>(new DistTuple<Ts...>(std::make_tuple(std::forward<Ts>(dists)...))) ),
    _num_dim(handle->num_dim()),
    _num_params(handle->num_params()),
    _lbound(handle->lbound()),
    _ubound(handle->ubound())
{ }

template<class RngT>
template<class... Ts>
CompositeDist<RngT>::CompositeDist(std::tuple<Ts...>&& dist_tuple) : 
    handle( std::unique_ptr<DistTupleHandle>(new DistTuple<Ts...>(std::move(dist_tuple))) ),
    _num_dim(handle->num_dim()),
    _num_params(handle->num_params()),
    _lbound(handle->lbound()),
    _ubound(handle->ubound())
{ }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::initialize(Ts&&... dists)
{
    handle = std::unique_ptr<DistTupleHandle>( new DistTuple<Ts...>(std::make_tuple(std::forward<Ts>(dists)...)) );
    _num_dim = handle->num_dim();
    _num_params = handle->num_params();
    _lbound = handle->lbound();
    _ubound = handle->ubound();    
}

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::initialize(std::tuple<Ts...>&& dist_tuple)
{
    handle = std::unique_ptr<DistTupleHandle>(new DistTuple<Ts...>(std::move(dist_tuple)));
    _num_dim = handle->num_dim();
    _num_params = handle->num_params();
    _lbound = handle->lbound();
    _ubound = handle->ubound();    
}


template<class RngT>
TypeInfoVecT CompositeDist<RngT>::component_types() const 
{ return handle->component_types(); }


template<class RngT>
IdxT CompositeDist<RngT>::num_component_dists() const
{return handle->num_dists();}

template<class RngT>
IdxT CompositeDist<RngT>::num_dim() const 
{return _num_dim;}

/** @brief Number of dims of each component as a vector */
template<class RngT>
UVecT CompositeDist<RngT>::components_num_dim() const 
{ return handle->components_num_dim(); }

template<class RngT>
StringVecT CompositeDist<RngT>::dim_variables() const 
{ return handle->dim_variables(); }

template<class RngT>
void CompositeDist<RngT>::set_dim_variables(const StringVecT &vars) 
{ 
    if(vars.size() != num_dim()){
        std::ostringstream msg;
        msg<<"Got bad dim variables vector size:"<<vars.size()<<" expected:"<<num_dim();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_dim_variables(vars); 
}

template<class RngT>
const VecT& CompositeDist<RngT>::lbound() const 
{ return _lbound; }

template<class RngT>
const VecT& CompositeDist<RngT>::ubound() const 
{ return _ubound; }

template<class RngT>
void CompositeDist<RngT>::set_lbound(const VecT &new_bound)
{ 
    if(new_bound.n_elem != num_dim()) {
        std::ostringstream msg;
        msg<<"Got bad lbound vector size:"<<new_bound.n_elem<<" expected:"<<num_dim();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_bounds(new_bound, ubound()); 
    update_bounds(); //Bounds may have changed with parameters
}

template<class RngT>
void CompositeDist<RngT>::set_ubound(const VecT &new_bound)
{ 
    if(new_bound.n_elem != num_dim()) {
        std::ostringstream msg;
        msg<<"Got bad ubound vector size:"<<new_bound.n_elem<<" expected:"<<num_dim();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_bounds(lbound(), new_bound); 
    update_bounds(); //Bounds may have changed with parameters
}

template<class RngT>
void CompositeDist<RngT>::set_bounds(const VecT &new_lbound,const VecT &new_ubound)
{ 
    if(new_lbound.n_elem != num_dim()) {
        std::ostringstream msg;
        msg<<"Got bad lbound vector size:"<<new_lbound.n_elem<<" expected:"<<num_dim();
        throw PriorHessianError("ParameterError",msg.str());
    }
    if(new_ubound.n_elem != num_dim()) {
        std::ostringstream msg;
        msg<<"Got bad ubound vector size:"<<new_ubound.n_elem<<" expected:"<<num_dim();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_bounds(new_lbound, new_ubound); 
    update_bounds(); //Bounds may have changed with parameters
}


/** @brief Strict bounds check for u withing the lbound and ubound */
template<class RngT>
bool CompositeDist<RngT>::in_bounds(const VecT &u) const
{ return arma::all(_lbound<u && u<_ubound); }

template<class RngT>
IdxT CompositeDist<RngT>::num_params() const 
{ return _num_params; }

template<class RngT>
UVecT CompositeDist<RngT>::components_num_params() const 
{ return handle->components_num_params(); }

template<class RngT>
VecT CompositeDist<RngT>::params() const 
{ return handle->params(); } 

template<class RngT>
void CompositeDist<RngT>::set_params(const VecT &new_params) 
{
    if(new_params.n_elem != num_params()) {
        std::ostringstream msg;
        msg<<"Got bad params vector size:"<<new_params.n_elem<<" expected:"<<num_params();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_params(new_params);
}    

template<class RngT>
StringVecT CompositeDist<RngT>::params_desc() const 
{ return handle->params_desc(); }

template<class RngT>
void CompositeDist<RngT>::set_params_desc(const StringVecT &new_desc) 
{ 
    if(new_desc.size() != num_params()) {
        std::ostringstream msg;
        msg<<"Got bad params desc vector size:"<<new_desc.size()<<" expected:"<<num_params();
        throw PriorHessianError("ParameterError",msg.str());
    }
    handle->set_params_desc(new_desc); 
}

//Functions mapped over underlying distributions
template<class RngT>
double CompositeDist<RngT>::cdf(const VecT &u) const 
{ return handle->cdf(u); }

template<class RngT>
double CompositeDist<RngT>::pdf(const VecT &u) const 
{ return handle->pdf(u); }    

template<class RngT>
double CompositeDist<RngT>::llh(const VecT &u) const 
{ return handle->llh(u); }

template<class RngT>
double CompositeDist<RngT>::rllh(const VecT &u) const 
{ return handle->rllh(u); }

template<class RngT>
VecT CompositeDist<RngT>::grad(const VecT &u) const 
{
    VecT g(_num_dim, arma::fill::zeros);
    handle->grad_accumulate(u,g);
    return g;
}

template<class RngT>
VecT CompositeDist<RngT>::grad2(const VecT &u) const
{
    VecT g2(_num_dim, arma::fill::zeros);
    handle->grad2_accumulate(u,g2);
    return g2;
}

template<class RngT>
MatT CompositeDist<RngT>::hess(const VecT &u) const 
{
    MatT h(_num_dim,_num_dim,arma::fill::zeros);
    handle->hess_accumulate(u,h);
    return h;
}
    
template<class RngT>
void CompositeDist<RngT>::grad_accumulate(const VecT &theta, VecT &grad) const 
{ return handle->grad_accumulate(theta,grad); }

template<class RngT>
void CompositeDist<RngT>::grad2_accumulate(const VecT &theta, VecT &grad2) const 
{ return handle->grad2_accumulate(theta,grad2); }

template<class RngT>
void CompositeDist<RngT>::hess_accumulate(const VecT &theta, MatT &hess) const 
{ return handle->hess_accumulate(theta,hess); }

template<class RngT>
void CompositeDist<RngT>::grad_grad2_accumulate(const VecT &theta, VecT &grad, VecT &grad2) const 
{ return handle->grad_grad2_accumulate(theta,grad,grad2); }

template<class RngT>
void CompositeDist<RngT>::grad_hess_accumulate(const VecT &theta, VecT &grad, MatT &hess) const 
{ return handle->grad_hess_accumulate(theta,grad,hess); }

template<class RngT>
VecT CompositeDist<RngT>::sample(RngT &rng) 
{ return handle->sample(rng); }

template<class RngT>
MatT CompositeDist<RngT>::sample(RngT &rng, IdxT num_samples) 
{ return handle->sample(rng,num_samples); }

template<class RngT>
VecT CompositeDist<RngT>::llh_components(const VecT &u) const 
{ return handle->llh_components(u); }

template<class RngT>
VecT CompositeDist<RngT>::rllh_components(const VecT &u) const 
{ return handle->rllh_components(u); }


template<class RngT>
void CompositeDist<RngT>::update_bounds()
{
    _lbound = handle->lbound();
    _ubound = handle->ubound();
}


/* CompositeDist<RngT>::DistTuple<Ts...> template methods */


template<class RngT>
template<class... Ts>
CompositeDist<RngT>::DistTuple<Ts...>::DistTuple(std::tuple<Ts...>&& dists) :
    dists(std::move(dists))
{}


template<class RngT>
template<class... Ts>
constexpr IdxT CompositeDist<RngT>::DistTuple<Ts...>::num_dists() const
{ return _num_dists; }

template<class RngT>
template<class... Ts>
TypeInfoVecT CompositeDist<RngT>::DistTuple<Ts...>::component_types() const
{ 
    return {std::type_index(typeid(Ts))...}; 
}

template<class RngT>
template<class... Ts>
constexpr IdxT CompositeDist<RngT>::DistTuple<Ts...>::num_dim() const
{
    return _num_dim;
}

template<class RngT>
template<class... Ts>
UVecT CompositeDist<RngT>::DistTuple<Ts...>::components_num_dim() const
{
    return {Ts::num_dim()...};
}

template<class RngT>
template<class... Ts>
StringVecT CompositeDist<RngT>::DistTuple<Ts...>::dim_variables() const
{
    StringVecT vars(_num_dim);
    append_dim_variables(vars.begin(),IndexT());
    return vars;
}

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::set_dim_variables(const StringVecT &vars)
{ 
    set_dim_variables(vars.begin(),IndexT()); 
}

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::lbound() const
{
    VecT lb(_num_dim);
    append_lbound(lb.begin(),IndexT());
    return lb;
}

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::ubound() const
{
    VecT ub(_num_dim);
    append_ubound(ub.begin(),IndexT());
    return ub;
}

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::set_bounds(const VecT &new_lbound, const VecT &new_ubound)
{ set_bounds(new_lbound.begin(),new_ubound.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
constexpr IdxT CompositeDist<RngT>::DistTuple<Ts...>::num_params() const
{ return _num_params; }

template<class RngT>
template<class... Ts>
UVecT CompositeDist<RngT>::DistTuple<Ts...>::components_num_params() const
{ return {Ts::num_params()...}; }        

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::params() const
{
    VecT params(_num_params);
    append_params(params.begin(),IndexT());
    return params;
}

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::set_params(const VecT &params)
{ set_params(params.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
StringVecT CompositeDist<RngT>::DistTuple<Ts...>::params_desc() const
{
    StringVecT desc(_num_params);
    append_params_desc(desc.begin(),IndexT());
    return desc;
}

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::set_params_desc(const StringVecT &desc)
{ set_params_desc(desc.begin(),IndexT()); }
        
template<class RngT>
template<class... Ts>
double CompositeDist<RngT>::DistTuple<Ts...>::cdf(const VecT &u) const
{ return cdf(u.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
double CompositeDist<RngT>::DistTuple<Ts...>::pdf(const VecT &u) const
{ return pdf(u.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
double CompositeDist<RngT>::DistTuple<Ts...>::llh(const VecT &u) const
{ return llh(u.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
double CompositeDist<RngT>::DistTuple<Ts...>::rllh(const VecT &u) const
{ return rllh(u.begin(),IndexT()); }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::grad_accumulate(const VecT &u, VecT &g) const
{ grad_accumulate(u,g,IndexT()); }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::grad2_accumulate(const VecT &u, VecT &g2) const
{ grad2_accumulate(u,g2,IndexT()); }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::hess_accumulate(const VecT &u, MatT &h) const
{ hess_accumulate(u,h,IndexT()); }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2) const
{ grad_grad2_accumulate(u,g,g2,IndexT()); }

template<class RngT>
template<class... Ts>
void CompositeDist<RngT>::DistTuple<Ts...>::grad_hess_accumulate(const VecT &u, VecT &g, MatT &h) const
{ grad_hess_accumulate(u,g,h,IndexT()); }

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::sample(RngT &rng)
{
    VecT s(_num_dim);
    sample(rng, s.begin(), IndexT());
    return s;
}

template<class RngT>
template<class... Ts>
MatT CompositeDist<RngT>::DistTuple<Ts...>::sample(RngT &rng, IdxT nSamples)
{
    MatT s(_num_dim,nSamples);
    sample(rng, s.begin(), nSamples, IndexT());
    return s;
}

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::llh_components(const VecT &theta) const
{ return llh_components(theta.begin(), IndexT());}

template<class RngT>
template<class... Ts>
VecT CompositeDist<RngT>::DistTuple<Ts...>::rllh_components(const VecT &theta) const
{ return rllh_components(theta.begin(), IndexT());}

/* CompositeDist<RngT>::DistTuple<Ts...> private template methods
 * These are variadic templates over the indexes and do the actual calls to the component dists
 */


template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::append_dim_variables(IterT v, std::index_sequence<I...>) const
{ 
    meta::call_in_order( {(std::get<I>(dists).append_var_name(v),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::set_dim_variables(IterT v,std::index_sequence<I...>)
{ 
    meta::call_in_order( {(std::get<I>(dists).set_var_name_iter(v),0)...} ); 
}  
        
template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::append_lbound(IterT p, std::index_sequence<I...>) const
{ 
    meta::call_in_order( {(std::get<I>(dists).append_lbound(p),0)...} ); 
}


template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::append_ubound(IterT p, std::index_sequence<I...>) const
{ 
    meta::call_in_order( {(std::get<I>(dists).append_ubound(p),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::set_bounds(IterT lb, IterT ub,std::index_sequence<I...>)
{ 
    meta::call_in_order( {(std::get<I>(dists).set_bounds_from_iter(lb,ub),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::append_params(IterT p, std::index_sequence<I...>) const
{ 
    meta::call_in_order( {(std::get<I>(dists).append_params(p),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::set_params(IterT p,std::index_sequence<I...>)
{ 
    meta::call_in_order( {(std::get<I>(dists).set_params_iter(p),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::append_params_desc(IterT p, std::index_sequence<I...>) const
{ 
    meta::call_in_order( {(std::get<I>(dists).append_params_desc(p),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::set_params_desc(IterT p,std::index_sequence<I...>)
{ 
    meta::call_in_order( {(std::get<I>(dists).set_params_desc(p),0)...} ); 
}
        
template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
double CompositeDist<RngT>::DistTuple<Ts...>::cdf(IterT u,std::index_sequence<I...>) const 
{ 
    return meta::prod_in_order( {std::get<I>(dists).cdf_from_iter(u)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
double CompositeDist<RngT>::DistTuple<Ts...>::pdf(IterT u,std::index_sequence<I...>) const 
{ 
    return meta::prod_in_order( {std::get<I>(dists).pdf_from_iter(u)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
double CompositeDist<RngT>::DistTuple<Ts...>::llh(IterT u,std::index_sequence<I...>) const 
{ 
    return meta::sum_in_order( {std::get<I>(dists).llh_from_iter(u)...} ); 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
double CompositeDist<RngT>::DistTuple<Ts...>::rllh(IterT u,std::index_sequence<I...>) const 
{ 
    return meta::sum_in_order( {std::get<I>(dists).rllh_from_iter(u)...} ); 
}

template<class RngT>
template<class... Ts>
template<std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::grad_accumulate(const VecT &u, VecT &g,std::index_sequence<I...>) const
{ 
    IdxT k=0;
    meta::call_in_order( {(std::get<I>(dists).grad_accumulate_idx(u,g,k),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::grad2_accumulate(const VecT &u, VecT &g2,std::index_sequence<I...>) const
{ 
    IdxT k=0;
    meta::call_in_order( {(std::get<I>(dists).grad2_accumulate_idx(u,g2,k),0)...} ); 
}

template<class RngT>
template<class... Ts>
template<std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::hess_accumulate(const VecT &u, MatT &m, std::index_sequence<I...>) const
{
    IdxT k=0;
    meta::call_in_order( {(std::get<I>(dists).hess_accumulate_idx(u,m,k),0)...} );
}

template<class RngT>
template<class... Ts>
template<std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::grad_grad2_accumulate(const VecT &u, VecT &g, VecT &g2, std::index_sequence<I...>) const
{ 
    IdxT k=0;
    meta::call_in_order( {(std::get<I>(dists).grad_grad2_accumulate_idx(u,g,g2,k),0)...} );
}

template<class RngT>
template<class... Ts>
template<std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::grad_hess_accumulate(const VecT &u, VecT &g, MatT &h,std::index_sequence<I...>) const
{
    IdxT k=0;
    meta::call_in_order( {(std::get<I>(dists).grad_hess_accumulate_idx(u,g,h,k),0)...} );
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::sample(RngT &rng, IterT s, std::index_sequence<I...>)
{
    meta::call_in_order( {(std::get<I>(dists).append_sample(rng,s),0)...} );
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
void CompositeDist<RngT>::DistTuple<Ts...>::sample(RngT &rng, IterT s, IdxT nSamples,std::index_sequence<I...>)
{
    for(IdxT n=0; n<nSamples; n++) 
        meta::call_in_order( {(std::get<I>(dists).append_sample(rng,s),0)...} );
}


template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
VecT CompositeDist<RngT>::DistTuple<Ts...>::llh_components(IterT theta,std::index_sequence<I...>) const 
{ 
    return VecT{std::get<I>(dists).llh_from_iter(theta)...}; 
}

template<class RngT>
template<class... Ts>
template<class IterT, std::size_t... I> 
VecT CompositeDist<RngT>::DistTuple<Ts...>::rllh_components(IterT theta,std::index_sequence<I...>) const 
{ 
    return VecT{std::get<I>(dists).rllh_from_iter(theta)...}; 
}



template<class RngT>
std::ostream& operator<<(std::ostream &out,const CompositeDist<RngT> &comp_dist)
{
    out<<"[CompositeDist]:\n";
    out<<"  NumComponentDists:"<<comp_dist.num_component_dists()<<"\n";
    out<<"  NumDim:"<<comp_dist.num_dim()<<"\n";
    out<<"  ComponentNumDim:"<<comp_dist.components_num_dim().t();
    out<<"  LBound:"<<comp_dist.lbound().t();
    out<<"  UBound:"<<comp_dist.ubound().t();
    auto vars=comp_dist.dim_variables();
    out<<"  Vars:[";
    for(auto v: vars) out<<v<<",";
    out<<"]\n";

    out<<"  NumParams:"<<comp_dist.num_params()<<"\n";
    out<<"  ComponentNumParams:"<<comp_dist.components_num_params().t();
    out<<"  Params:"<<comp_dist.params().t();
    auto param_desc=comp_dist.params_desc();
    out<<"  ParamDesc:[";
    for(auto &v: param_desc) out<<v<<",";
    out<<"]\n";
    
    return out;
}


} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_COMPOSITEDIST_H */
