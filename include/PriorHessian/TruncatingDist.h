/** @file TruncatingDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief TruncatingDist, and subclasses InfiniteDist and SemiInfiniteDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_TRUNCATINGDIST_H
#define _PRIOR_HESSIAN_TRUNCATINGDIST_H

#include "PriorHessian/UnivariateDist.h"

namespace prior_hessian {

/** @brief Abstract Distribution that allows truncation.
 * 
 * sunclass must implement functions cdf_un
 * 
 */
template<class Derived>
class TruncatingDist : public UnivariateDist<Derived>
{
protected:
    TruncatingDist(std::string var_name, StringVecT&& param_desc);
public:
    
    double cdf(double x) const;
    double pdf(double x) const;
    double icdf(double u) const;
    double llh(double x) const;

protected:
    bool truncated=false;
   
    double lbound_cdf; // cdf(_lbound)
    double bounds_cdf_delta; // (cdf(_ubound) - cdf(_lbound))
    double llh_truncation_const;
    double compute_llh_truncation_const() const;
};

template<class Derived>
class InfiniteDist : public TruncatingDist<Derived>
{
public:
    InfiniteDist(std::string var_name, StringVecT&& param_desc);
    InfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
    void set_lbound(double lbound);    
    void set_ubound(double ubound);    
};

template<class Derived>
class SemiInfiniteDist : public TruncatingDist<Derived>
{
public:
    SemiInfiniteDist(std::string var_name, StringVecT&& param_desc);
    SemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
    void set_lbound(double lbound);    
    void set_ubound(double ubound);    
};

template<class Derived>
class PositiveSemiInfiniteDist : public TruncatingDist<Derived>
{
public:
    PositiveSemiInfiniteDist(std::string var_name, StringVecT&& param_desc);
    PositiveSemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
    void set_lbound(double lbound);    
    void set_ubound(double ubound);    
};


template<class Derived>
TruncatingDist<Derived>::TruncatingDist(std::string var_name, StringVecT&& param_desc) : 
    UnivariateDist<Derived>(var_name, std::move(param_desc)) 
{ }
        
template<class Derived>
double TruncatingDist<Derived>::cdf(double x) const
{
    double val = static_cast<Derived const*>(this)->unbounded_cdf(x);
    if(truncated) val = (val - lbound_cdf) / bounds_cdf_delta;
    return val;
}

template<class Derived>
double TruncatingDist<Derived>::icdf(double u) const
{
    if(truncated) u = lbound_cdf + u*bounds_cdf_delta; //Transform u if necessary
    return static_cast<Derived const*>(this)->unbounded_icdf(u);
}

template<class Derived>
double TruncatingDist<Derived>::pdf(double x) const
{
    double val = static_cast<Derived const*>(this)->unbounded_pdf(x);
    if(truncated) val/=bounds_cdf_delta;
    return val;
}

template<class Derived>
double TruncatingDist<Derived>::llh(double x) const
{
    return static_cast<Derived const*>(this)->rllh(x) + this->llh_const + llh_truncation_const;
}


template<class Derived>
double TruncatingDist<Derived>::compute_llh_truncation_const() const
{
    return truncated ? 0 : -log(bounds_cdf_delta);
}




template<class Derived>
InfiniteDist<Derived>::InfiniteDist(std::string var_name, StringVecT&& param_desc) :
    InfiniteDist(-INFINITY,INFINITY,var_name,std::move(param_desc))
{ }

template<class Derived>
InfiniteDist<Derived>::InfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ }

/**
 * Uses unbounded_cdf from subclass, so must only be called after construction of subclass...
 * 
 */
template<class Derived>
void InfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    UnivariateDist<Derived>::set_bounds(lbound,ubound); //Set top-level bounds
    
    this->truncated = (-INFINITY<lbound || ubound<INFINITY); //This tells Base dist if the bounds are truncating or not.
    if(lbound == -INFINITY) {
        this->lbound_cdf = 0;
    } else {
        this->lbound_cdf = static_cast<Derived const*>(this)->unbounded_cdf(lbound); //F(lb)
    }
    if(ubound == INFINITY) {
        this->bounds_cdf_delta = 1-this->lbound_cdf;
    } else {
        this->bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound) - this->lbound_cdf; // F(ub)-F(lb)
    }
    this->llh_truncation_const = this->compute_llh_truncation_const();
}

template<class Derived>
void InfiniteDist<Derived>::set_lbound(double lbound)
{
    set_bounds(lbound,this->ubound());
}

template<class Derived>
void InfiniteDist<Derived>::set_ubound(double ubound)
{
    set_bounds(this->lbound(),ubound);
}

template<class Derived>
SemiInfiniteDist<Derived>::SemiInfiniteDist(std::string var_name, StringVecT&& param_desc) :
    SemiInfiniteDist(0,INFINITY, var_name, std::move(param_desc))
{ }

template<class Derived>
SemiInfiniteDist<Derived>::SemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ }

/**
 * Uses unbounded_cdf from subclass, so must only be called after construction of subclass...
 * 
 */
template<class Derived>
void SemiInfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound<0) {
        std::ostringstream msg;
        msg<<"SemiInfinite: lbound must be positive. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    } 
    UnivariateDist<Derived>::set_bounds(lbound,ubound); //Set top-level bounds
    
    this->truncated = (lbound>0 || ubound<INFINITY); //This tells Base dist if the bounds are truncating or not.
    
    if(lbound==0){
        this->lbound_cdf = 0;
    } else {
        this->lbound_cdf = static_cast<Derived const*>(this)->unbounded_cdf(lbound); //F(lb)
    }
    
    if(ubound==INFINITY) {
        this->bounds_cdf_delta = 1-this->lbound_cdf;
    } else {
        this->bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound) - this->lbound_cdf; // F(ub)-F(lb)
    }
    this->llh_truncation_const = this->compute_llh_truncation_const();
}

template<class Derived>
void SemiInfiniteDist<Derived>::set_lbound(double lbound)
{
    set_bounds(lbound,this->ubound());
}

template<class Derived>
void SemiInfiniteDist<Derived>::set_ubound(double ubound)
{
    set_bounds(this->lbound(),ubound);
}

template<class Derived>
PositiveSemiInfiniteDist<Derived>::PositiveSemiInfiniteDist(std::string var_name, StringVecT&& param_desc) :
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ }

template<class Derived>
PositiveSemiInfiniteDist<Derived>::PositiveSemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ }

/**
 * Uses unbounded_cdf from subclass, so must only be called after construction of subclass...
 * 
 */
template<class Derived>
void PositiveSemiInfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound<=0) {
        std::ostringstream msg;
        msg<<"PositiveDist: lbound must be positive and non-zero. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    } 

    UnivariateDist<Derived>::set_bounds(lbound,ubound); //Set top-level bounds
    
    this->truncated = ubound<INFINITY; //This tells Base dist if the bounds are truncating or not.
    this->lbound_cdf = 0; //F(lb)=0 by definition for Positive distributions like the Pareto
    
    if(ubound==INFINITY) {
        this->bounds_cdf_delta = 1;
    } else {
        this->bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound); // F(ub)
    }
    this->llh_truncation_const = this->compute_llh_truncation_const();
}

template<class Derived>
void PositiveSemiInfiniteDist<Derived>::set_lbound(double lbound)
{
    set_bounds(lbound,this->ubound());
}

template<class Derived>
void PositiveSemiInfiniteDist<Derived>::set_ubound(double ubound)
{
    set_bounds(this->lbound(),ubound);
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_TRUNCATINGDIST_H */
