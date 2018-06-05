/** @file UnivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief UnivariateDist base class.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_UNIVARIATEDIST_H
#define _PRIOR_HESSIAN_UNIVARIATEDIST_H

#include "PriorHessian/BaseDist.h"

namespace prior_hessian {
    
template<class Derived>
class UnivariateDist : public BaseDist {
    template<class RngT> friend class CompositeDist;
public:
    UnivariateDist(std::string var_name, StringVecT &&params_desc);
    /* Var name and dimensionality */
    constexpr static IdxT num_dim();
    const std::string& var_name();
    void set_var_name(std::string var_name);

    /* Bounds */
    double get_lbound() const; 
    double get_ubound() const;
    VecT get_params() const;
    void set_params(const VecT& p);
    
    template<class RngT> double sample(RngT &rng);
protected:
    /* Helper methods for use by CompositeDist */
    template<class IterT> void append_var_name(IterT &v) const;
    template<class IterT> void set_var_name(IterT &v);
    template<class IterT> void append_lbound(IterT &v) const;
    template<class IterT> void append_ubound(IterT &v) const;
    template<class IterT> void set_bounds_from_iter(IterT& lbounds, IterT &ubounds);   
    
    /* Univariate helper operations */
    template<class IterT> double cdf_from_iter(IterT &u) const;
    template<class IterT> double pdf_from_iter(IterT &u) const;
    template<class IterT> double llh_from_iter(IterT &u) const;
    template<class IterT> double rllh_from_iter(IterT &u) const;

    /* Sample helper operatations */
    template<class RngT, class IterT> void append_sample(RngT &rng, IterT &iter);

    /* Vector and matrix helper operations */
    void grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const;
    void grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const; 
    void hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const; 
    void grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const; 
    void grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const; 

    std::string _var_name;
    double _lbound;
    double _ubound;
    double llh_const;
    

    void set_bounds(double lbound, double ubound);
    void set_lbound(double lbound);
    void set_ubound(double ubound);
};

template<class Derived>
UnivariateDist<Derived>::UnivariateDist(std::string var_name, StringVecT &&params_desc) :
    BaseDist(std::move(params_desc)),
    _var_name(var_name)
{ }

template<class Derived>
constexpr IdxT UnivariateDist<Derived>::num_dim()
{ return 1; }

template<class Derived>
const std::string& UnivariateDist<Derived>::var_name() 
{ return _var_name; }

template<class Derived>
void UnivariateDist<Derived>::set_var_name(std::string var_name) 
{ _var_name = var_name;}


template<class Derived>
double UnivariateDist<Derived>::get_lbound() const 
{ return _lbound; }

template<class Derived>
double UnivariateDist<Derived>::get_ubound() const 
{ return _ubound; }

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::append_lbound(IterT& p) const 
{ *p++ = _lbound;} 

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::append_ubound(IterT& p) const 
{ *p++ = _ubound;} 

/* params */
template<class Derived>
VecT UnivariateDist<Derived>::get_params() const 
{ 
    VecT p(num_params());
    auto start = p.begin();
    static_cast<const Derived*>(this)->append_params(start);
    return p;
}

template<class Derived>
void UnivariateDist<Derived>::set_params(const VecT& p) 
{ 
    auto start = p.cbegin();
    static_cast<Derived*>(this)->set_params(start); 
}     


template<class Derived>
void UnivariateDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound>=ubound) {
        std::ostringstream msg;
        msg<<"UnivariateDist: lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _lbound = lbound;
    _ubound = ubound;
}

template<class Derived>
void UnivariateDist<Derived>::set_lbound(double lbound)
{
    
    if(lbound >= _ubound) {
        std::ostringstream msg;
        msg<<"UnivariateDist: lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<_ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _lbound = lbound;
}

template<class Derived>
void UnivariateDist<Derived>::set_ubound(double ubound)
{
    
    if(_lbound >= ubound) {
        std::ostringstream msg;
        msg<<"UnivariateDist: lbound must be smaller than ubound. Got: L:"<<_lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    _ubound = ubound;
}

template<class Derived>
template<class RngT> 
double  UnivariateDist<Derived>::sample(RngT &rng)
{
    UniformDistT uniform;
    return  static_cast<Derived const*>(this)->icdf(uniform(rng)); //sample via iCDF method
}


template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::set_var_name(IterT& v)
{ _var_name = *v++; } 

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::append_var_name(IterT& v) const 
{ *v++ = _var_name; } 

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::set_bounds_from_iter(IterT& lbound, IterT& ubound)
{
    return static_cast<Derived*>(this)->set_bounds(*lbound++, *ubound++); //call top-level set_bounds first
}

/* Univariate operations */
template<class Derived>
template<class IterT>
double UnivariateDist<Derived>::cdf_from_iter(IterT &u) const
{ return static_cast<Derived const*>(this)->cdf(*u++); }

template<class Derived>
template<class IterT>
double UnivariateDist<Derived>::pdf_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->pdf(*u++); }

template<class Derived>
template<class IterT>
double UnivariateDist<Derived>::llh_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->llh(*u++); }

template<class Derived>
template<class IterT>
double UnivariateDist<Derived>::rllh_from_iter(IterT &u) const 
{ return static_cast<Derived const*>(this)->rllh(*u++); }


/* Vector and matrix operations */
template<class Derived>
void UnivariateDist<Derived>::grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const 
{ 
    g(k) += static_cast<Derived const*>(this)->grad(u(k));
    k++;
}

template<class Derived>
void UnivariateDist<Derived>::grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const 
{ 
    g2(k) += static_cast<Derived const*>(this)->grad2(u(k));
    k++;
}

template<class Derived>
void UnivariateDist<Derived>::hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const 
{ 
    h(k,k) += static_cast<Derived const*>(this)->grad2(u(k));
    k++;
}
    
template<class Derived>
void UnivariateDist<Derived>::grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const 
{ 
    static_cast<Derived const*>(this)->grad_grad2_accumulate(u(k),g(k),g2(k));
    k++;
}

template<class Derived>
void UnivariateDist<Derived>::grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const 
{ 
    static_cast<Derived const*>(this)->grad_grad2_accumulate(u(k),g(k),h(k,k));
    k++;
}

/* Sampling */
template<class Derived>
template<class RngT, class IterT> 
void UnivariateDist<Derived>::append_sample(RngT &rng, IterT &iter)
{
    *iter++ = static_cast<Derived*>(this)->sample(rng);
}


} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_UNIVARIATEDIST_H */
