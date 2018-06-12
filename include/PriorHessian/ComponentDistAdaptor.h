/** @file ComponentDistAdaptor.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief ComponentDistAdaptor base class.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_COMPONENTDISTADAPTOR_H
#define _PRIOR_HESSIAN_COMPONENTDISTADAPTOR_H

#include "PriorHessian/util.h"
#include "PriorHessian/PriorHessianError.h"

namespace prior_hessian {

class BaseComponentDistAdaptor {
};

//Forward decl
template<class RngT>
class CompositeDist;

template<class Dist>
class ComponentDistAdaptor : public BaseComponentDistAdaptor {   
    template<class RngT> friend class CompositeDist;
public:
    ComponentDistAdaptor();
    ComponentDistAdaptor(Dist dist, std::string var_name);
    ComponentDistAdaptor(Dist dist, std::string var_name, StringVecT &&param_names);
    
    template<class T>
    std::
    T make_adaptor(Dist dist, std::string var_name);
    
    
    IdxT num_dim();
    const StringVecT& var_names() const;
    const std::string& var_name(int i) const;
    void set_var_names(const StringVecT &var_name);
    void set_var_name(int i, std::string new_var_name);
    
    IdxT num_params() const;
    const StringVecT& param_names() const;
    const StringVecT& param_name(int i) const;
    void set_param_names(const StringVecT &names);
    void set_param_name(int i, const std::string &name);
    
    Dist& get_dist();
    const Dist& get_dist() const;
    Dist& set_dist(Dist dist);
    Dist& set_dist(Dist dist, std::string var_name);
    Dist& set_dist(Dist dist, std::string var_name, StringVecT &&param_names);
    
protected:
    Dist dist;
    IdxT _num_dim;
    StringVecT _var_names;
    StringVecT _param_names;
    
    template<class IterT> void append_var_name(IterT &v) const;
    template<class IterT> void set_var_name_iter(IterT &v);

    template<class IterT> void append_lbound(IterT &v) const;
    template<class IterT> void append_ubound(IterT &v) const;
    template<class IterT> void set_bounds_from_iter(IterT& lbounds, IterT &ubounds);   
    
    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params_iter(IterT& p);   

    template<class IterT> void append_param_names(IterT& p) const;
    template<class IterT> void set_param_names_iter(IterT& d); 

    template<class IterT> double cdf_from_iter(IterT &u) const;
    template<class IterT> double pdf_from_iter(IterT &u) const;
    template<class IterT> double llh_from_iter(IterT &u) const;
    template<class IterT> double rllh_from_iter(IterT &u) const;

    void grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const;
    void grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const; 
    void hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const; 
    void grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const; 
    void grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const; 

    template<class RngT, class IterT> void append_sample(RngT &rng, IterT &iter);
};


ComponentDistAdaptor();
ComponentDistAdaptor(Dist dist, std::string var_name);
ComponentDistAdaptor(Dist dist, std::string var_name, StringVecT &&param_names);

template<class Derived>
ComponentDistAdaptor<Derived>::ComponentDistAdaptor(StringVecT var_names, StringVecT &&param_names) 
    : BaseComponentDistAdaptor(),
    _var_name(var_name)
{ }

template<class Derived>
constexpr IdxT ComponentDistAdaptor<Derived>::num_dim()
{ return 1; }

template<class Derived>
const std::string& ComponentDistAdaptor<Derived>::var_name() 
{ return _var_name; }

template<class Derived>
void ComponentDistAdaptor<Derived>::set_var_name(std::string var_name) 
{ _var_name = var_name;}


template<class Derived>
double ComponentDistAdaptor<Derived>::get_lbound() const 
{ return _lbound; }

template<class Derived>
double ComponentDistAdaptor<Derived>::get_ubound() const 
{ return _ubound; }

template<class Derived>
template<class IterT>
void ComponentDistAdaptor<Derived>::append_lbound(IterT& p) const 
{ *p++ = _lbound;} 

template<class Derived>
template<class IterT>
void ComponentDistAdaptor<Derived>::append_ubound(IterT& p) const 
{ *p++ = _ubound;} 

/* params */
template<class Derived>
VecT ComponentDistAdaptor<Derived>::params() const 
{ 
    VecT p(num_params());
    auto start = p.begin();
    static_cast<const Derived*>(this)->append_params_iter(start);
    return p;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::set_params(const VecT& p) 
{ 
    auto start = p.cbegin();
    static_cast<Derived*>(this)->set_params_iter(start); 
}     


template<class Derived>
void ComponentDistAdaptor<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound>=ubound) {
        std::ostringstream msg;
        msg<<"ComponentDistAdaptor: lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _lbound = lbound;
    _ubound = ubound;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::set_lbound(double lbound)
{
    
    if(lbound >= _ubound) {
        std::ostringstream msg;
        msg<<"ComponentDistAdaptor: lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<_ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _lbound = lbound;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::set_ubound(double ubound)
{
    
    if(_lbound >= ubound) {
        std::ostringstream msg;
        msg<<"ComponentDistAdaptor: lbound must be smaller than ubound. Got: L:"<<_lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _ubound = ubound;
}

template<class Derived>
template<class RngT> 
double  ComponentDistAdaptor<Derived>::sample(RngT &rng)
{
    UniformDistT uniform;
    return  static_cast<Derived const*>(this)->icdf(uniform(rng)); //sample via iCDF method
}


template<class Derived>
template<class IterT>
void ComponentDistAdaptor<Derived>::set_var_name(IterT& v)
{ _var_name = *v++; } 

template<class Derived>
template<class IterT>
void ComponentDistAdaptor<Derived>::append_var_name(IterT& v) const 
{ *v++ = _var_name; } 

template<class Derived>
template<class IterT>
void ComponentDistAdaptor<Derived>::set_bounds_from_iter(IterT& lbound, IterT& ubound)
{
    return static_cast<Derived*>(this)->set_bounds(*lbound++, *ubound++); //call top-level set_bounds first
}

/* Univariate operations */
template<class Derived>
template<class IterT>
double ComponentDistAdaptor<Derived>::cdf_from_iter(IterT &u) const
{ return static_cast<Derived const*>(this)->cdf(*u++); }

template<class Derived>
template<class IterT>
double ComponentDistAdaptor<Derived>::pdf_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->pdf(*u++); }

template<class Derived>
template<class IterT>
double ComponentDistAdaptor<Derived>::llh_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->llh(*u++); }

template<class Derived>
template<class IterT>
double ComponentDistAdaptor<Derived>::rllh_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->rllh(*u++); }


/* Vector and matrix operations */
template<class Derived>
void ComponentDistAdaptor<Derived>::grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const 
{ 
    g(k) += static_cast<Derived const*>(this)->grad(u(k));
    k++;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const 
{ 
    g2(k) += static_cast<Derived const*>(this)->grad2(u(k));
    k++;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const 
{ 
    h(k,k) += static_cast<Derived const*>(this)->grad2(u(k));
    k++;
}
    
template<class Derived>
void ComponentDistAdaptor<Derived>::grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const 
{ 
    static_cast<Derived const*>(this)->grad_grad2_accumulate(u(k),g(k),g2(k));
    k++;
}

template<class Derived>
void ComponentDistAdaptor<Derived>::grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const 
{ 
    static_cast<Derived const*>(this)->grad_grad2_accumulate(u(k),g(k),h(k,k));
    k++;
}

/* Sampling */
template<class Derived>
template<class RngT, class IterT> 
void ComponentDistAdaptor<Derived>::append_sample(RngT &rng, IterT &iter)
{
    *iter++ = static_cast<Derived*>(this)->sample(rng);
}


} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_COMPONENTDISTADAPTOR_H */
