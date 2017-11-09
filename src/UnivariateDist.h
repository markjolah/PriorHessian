/** @file UnivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief UnivariateDist base class.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_UNIVARIATEDIST_H
#define _PRIOR_HESSIAN_UNIVARIATEDIST_H

#include "BaseDist.h"
namespace prior_hessian {
    
template<class Derived>
class UnivariateDist : public BaseDist {

public:
    UnivariateDist(double lbound, double ubound, std::string var_name, StringVecT &&params_desc);

    /* Var name and dimensionality */
    constexpr static IdxT num_dim();
    const std::string& var_name();
    void set_var_name(std::string var_name);
    template<class IterT> void set_var_name(IterT &v);
    template<class IterT> void insert_var_name(IterT &v) const;

    /* Bounds */
    double lbound() const;
    double ubound() const;
    template<class IterT> void insert_lbound(IterT &v) const;
    template<class IterT> void insert_ubound(IterT &v) const;

    /* Params */
    VecT params() const;
    void set_params(const VecT& p);
    
    /* Univariate operations */
    template<class IterT> double cdf_from_iter(IterT &u) const;
    template<class IterT> double pdf_from_iter(IterT &u) const;
    template<class IterT> double llh_from_iter(IterT &u) const;
    template<class IterT> double rllh_from_iter(IterT &u) const;
    
    /* Vector and matrix operations */
    void grad_accumulate_idx(const VecT &u, VecT &g, IdxT &k) const;
    void grad2_accumulate_idx(const VecT &u, VecT &g2, IdxT &k) const; 
    void hess_accumulate_idx(const VecT &u, MatT &h, IdxT &k) const; 
    void grad_grad2_accumulate_idx(const VecT &u, VecT &g, VecT &g2, IdxT &k) const; 
    void grad_hess_accumulate_idx(const VecT &u, VecT &g, MatT &h, IdxT &k) const; 
    template<class RngT, class IterT> void insert_sample(RngT &rng, IterT &iter);
protected:
    double _lbound, _ubound;
    std::string _var_name;
};

template<class Derived>
UnivariateDist<Derived>::UnivariateDist(double lbound, double ubound, std::string var_name, StringVecT &&params_desc) :
    BaseDist(std::move(params_desc)),
    _lbound(lbound), 
    _ubound(ubound), 
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
template<class IterT>
void UnivariateDist<Derived>::set_var_name(IterT& v)
{ _var_name = *v++; } 

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::insert_var_name(IterT& v) const 
{ *v++ = _var_name; } 

template<class Derived>
double UnivariateDist<Derived>::lbound() const 
{ return _lbound; }

template<class Derived>
double UnivariateDist<Derived>::ubound() const 
{ return _ubound; }

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::insert_lbound(IterT& p) const 
{ *p++ = _lbound;} 

template<class Derived>
template<class IterT>
void UnivariateDist<Derived>::insert_ubound(IterT& p) const 
{ *p++ = _ubound;} 

/* params */
template<class Derived>
VecT UnivariateDist<Derived>::params() const 
{ 
    VecT p(num_params());
    static_cast<Derived*>(this)->insert_params(p.begin());
    return p;
}

template<class Derived>
void UnivariateDist<Derived>::set_params(const VecT& p) 
{ static_cast<Derived*>(this)->set_params(p.cbegin()); }     

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
void UnivariateDist<Derived>::insert_sample(RngT &rng, IterT &iter)
{
    *iter++ = static_cast<Derived*>(this)->sample(rng);
}


} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_UNIVARIATEDIST_H */
