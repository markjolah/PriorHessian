/** @file TruncatingDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief TruncatingDist, and subclasses InfiniteDist and SemiInfiniteDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_TRUNCATINGDIST_H
#define _PRIOR_HESSIAN_TRUNCATINGDIST_H


#include "UnivariateDist.h"

namespace prior_hessian {

/** @brief Abstract Distribution that allows truncation.
 * 
 * sunclass must implement functions cdf_un
 * 
 */
template<class Derived>
class TruncatingDist : public UnivariateDist<Derived>
{
public:
    TruncatingDist(std::string var_name, StringVecT&& param_desc);
    
    double cdf(double x) const;
    double pdf(double x) const;
    double icdf(double u) const;

protected:
    bool truncated=false;
   
    double lbound_cdf; // cdf(_lbound)
    double bounds_cdf_delta; // (cdf(_ubound) - cdf(_lbound))
    
    double compute_llh_const() const;
};

template<class Derived>
class InfiniteDist : public TruncatingDist<Derived>
{
public:
    InfiniteDist(std::string var_name, StringVecT&& param_desc);
    InfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
};

template<class Derived>
class SemiInfiniteDist : public TruncatingDist<Derived>
{
public:
    SemiInfiniteDist(std::string var_name, StringVecT&& param_desc);
    SemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
};

template<class Derived>
class PositiveSemiInfiniteDist : public TruncatingDist<Derived>
{
public:
    PositiveSemiInfiniteDist(std::string var_name, StringVecT&& param_desc);
    PositiveSemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc); 

    void set_bounds(double lbound, double ubound);    
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
double TruncatingDist<Derived>::compute_llh_const() const
{
    double val = static_cast<Derived const*>(this)->compute_unbounded_llh_const();
    if(truncated) val -= log(bounds_cdf_delta);
    return val;
}




template<class Derived>
InfiniteDist<Derived>::InfiniteDist(std::string var_name, StringVecT&& param_desc) :
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(-INFINITY,INFINITY);
}

template<class Derived>
InfiniteDist<Derived>::InfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(lbound,ubound);
}

template<class Derived>
void InfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound>=ubound) {
        std::ostringstream msg;
        msg<<"lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    TruncatingDist<Derived>::truncated = (-INFINITY<lbound || ubound<INFINITY); //This tells Base dist if the bounds are truncating or not.
    
    if(lbound==-INFINITY){
        TruncatingDist<Derived>::lbound_cdf=0;
    } else {
        TruncatingDist<Derived>::lbound_cdf = static_cast<Derived const*>(this)->unbounded_cdf(lbound); //F(lb)
    }
    if(ubound==INFINITY) {
        TruncatingDist<Derived>::bounds_cdf_delta = 1-TruncatingDist<Derived>::lbound_cdf;
    } else {
        TruncatingDist<Derived>::bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound) 
                                                        - TruncatingDist<Derived>::lbound_cdf; // F(ub)-F(lb)
    }
    
    UnivariateDist<Derived>::llh_const = TruncatingDist<Derived>::compute_llh_const(); //Truncation terms.
    UnivariateDist<Derived>::_lbound = lbound;
    UnivariateDist<Derived>::_ubound = ubound;
}


template<class Derived>
SemiInfiniteDist<Derived>::SemiInfiniteDist(std::string var_name, StringVecT&& param_desc) :
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(0,INFINITY);
}

template<class Derived>
SemiInfiniteDist<Derived>::SemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(lbound,ubound);
}

template<class Derived>
void SemiInfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound<0) {
        std::ostringstream msg;
        msg<<"SemiInfinite: lbound must be positive. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    } else if(lbound>=ubound) {
        std::ostringstream msg;
        msg<<"lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    
    TruncatingDist<Derived>::truncated = (lbound>0 || ubound<INFINITY); //This tells Base dist if the bounds are truncating or not.
    
    if(lbound==0){
        TruncatingDist<Derived>::lbound_cdf = 0;
    } else {
        TruncatingDist<Derived>::lbound_cdf = static_cast<Derived const*>(this)->unbounded_cdf(lbound); //F(lb)
    }
    
    if(ubound==INFINITY) {
        TruncatingDist<Derived>::bounds_cdf_delta = 1-TruncatingDist<Derived>::lbound_cdf;
    } else {
        TruncatingDist<Derived>::bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound) - TruncatingDist<Derived>::lbound_cdf; // F(ub)-F(lb)
    }
    
    UnivariateDist<Derived>::llh_const = TruncatingDist<Derived>::compute_llh_const(); //Truncation terms.
    UnivariateDist<Derived>::_lbound = lbound;
    UnivariateDist<Derived>::_ubound = ubound;
}


template<class Derived>
PositiveSemiInfiniteDist<Derived>::PositiveSemiInfiniteDist(std::string var_name, StringVecT&& param_desc) :
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(1,INFINITY);
}

template<class Derived>
PositiveSemiInfiniteDist<Derived>::PositiveSemiInfiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    TruncatingDist<Derived>(var_name, std::move(param_desc))
{ 
    set_bounds(lbound,ubound);
}

template<class Derived>
void PositiveSemiInfiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound<=0) {
        std::ostringstream msg;
        msg<<"PositiveDist: lbound must be positive and non-zero. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    } else if(lbound>=ubound) {
        std::ostringstream msg;
        msg<<"lbound must be smaller than ubound. Got: L:"<<lbound<<" U:"<<ubound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    
    TruncatingDist<Derived>::truncated = ubound<INFINITY; //This tells Base dist if the bounds are truncating or not.
    TruncatingDist<Derived>::lbound_cdf = 0; //F(lb)=0 by definition for Positive distributions like the Pareto
    
    if(ubound==INFINITY) {
        TruncatingDist<Derived>::bounds_cdf_delta = 1;
    } else {
        TruncatingDist<Derived>::bounds_cdf_delta = static_cast<Derived const*>(this)->unbounded_cdf(ubound); // F(ub)
    }
    
    UnivariateDist<Derived>::llh_const = TruncatingDist<Derived>::compute_llh_const(); //Truncation terms.
    UnivariateDist<Derived>::_lbound = lbound;
    UnivariateDist<Derived>::_ubound = ubound;
}


} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_TRUNCATINGDIST_H */
