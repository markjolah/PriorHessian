/** @file TruncatedDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief TruncatedDist declaration and templated methods definitions
 * 
 */
#ifndef _PRIOR_HESSIAN_TRUNCATEDDIST_H
#define _PRIOR_HESSIAN_TRUNCATEDDIST_H

#include <cmath>

#include "PriorHessian/PriorHessianError.h"

namespace prior_hessian {

/** @brief 
 * 
 */
template<class Dist>
class TruncatedDist : public Dist
{
public:
    TruncatedDist();
    TruncatedDist(double lbound, double ubound);
    explicit TruncatedDist(const Dist &dist);
    explicit TruncatedDist(Dist &&dist);
    TruncatedDist(const Dist &dist, double lbound, double ubound);
    TruncatedDist(Dist &&dist, double lbound, double ubound);

    double lbound() const;
    double ubound() const;
    double global_lbound() const;
    double global_ubound() const;
    bool truncated() const;
    
    void set_bounds(double lbound, double ubound);    
    void set_lbound(double lbound);    
    void set_ubound(double ubound);    

    double cdf(double x) const;
    double pdf(double x) const;
    double icdf(double u) const;
    double llh(double x) const;

    template<class RngT>
    double sample(RngT &rng) const;
protected:
    double _truncated_lbound;
    double _truncated_ubound;
    bool _truncated = false;

    double lbound_cdf; // cdf(_lbound)
    double bounds_cdf_delta; // (cdf(_ubound) - cdf(_lbound))
    double llh_truncation_const;// -log(bounds_cdf_delta)   
};

template<class Dist>
TruncatedDist<Dist>::TruncatedDist() 
    : TruncatedDist(Dist{})
{ }

template<class Dist>
TruncatedDist<Dist>::TruncatedDist(double lbound, double ubound) 
    : TruncatedDist(Dist{}, lbound, ubound)
{ }

template<class Dist>
TruncatedDist<Dist>::TruncatedDist(const Dist &dist) 
    : TruncatedDist(dist, dist.lbound(), dist.ubound())
{ }

template<class Dist>
TruncatedDist<Dist>::TruncatedDist(Dist &&dist)
    : TruncatedDist(std::move(dist), dist.lbound(), dist.ubound())
{ }

template<class Dist>
TruncatedDist<Dist>::TruncatedDist(const Dist &dist, double lbound, double ubound)
    : Dist(dist)
{ set_bounds(lbound,ubound); }

template<class Dist>
TruncatedDist<Dist>::TruncatedDist(Dist &&dist, double lbound, double ubound)
    : Dist(std::move(dist))
{ set_bounds(lbound,ubound); }

template<class Dist>
double TruncatedDist<Dist>::lbound() const
{ return _truncated_lbound; }

template<class Dist>
double TruncatedDist<Dist>::ubound() const
{ return _truncated_ubound; }

template<class Dist>
double TruncatedDist<Dist>::global_lbound() const
{ return Dist::lbound(); }

template<class Dist>
double TruncatedDist<Dist>::global_ubound() const
{ return Dist::ubound(); }

template<class Dist>
bool TruncatedDist<Dist>::truncated() const
{ return _truncated; }

template<class Dist>
void TruncatedDist<Dist>::set_bounds(double lbound, double ubound)
{
    if( !(lbound >= global_lbound()) ) {   //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid lbound:"<<lbound<<" with regard to global_lbound:"<<global_lbound();
        throw ParameterValueError(msg.str());
    }
    if( !(ubound >= global_ubound()) ) {    //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<ubound<<" with regard to global_ubound:"<<global_ubound();
        throw ParameterValueError(msg.str());
    }
    if( !(lbound < ubound) ) { 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid bounds lbound:"<<lbound<<" >= ubound:"<<ubound;
        throw ParameterValueError(msg.str());
    }
    _truncated = (lbound>global_lbound() || ubound<global_ubound());
    if(_truncated) {
        lbound_cdf = (lbound==global_lbound()) ? 0 : Dist::cdf(lbound);
        double ubound_cdf = (ubound==global_ubound()) ? 1 : Dist::cdf(ubound);
        bounds_cdf_delta = ubound_cdf - lbound_cdf;
        llh_truncation_const = -log(bounds_cdf_delta);
    } else {
        lbound_cdf = 0;
        bounds_cdf_delta = 1;
        llh_truncation_const = 0;
    }
    _truncated_lbound = lbound;
    _truncated_ubound = ubound;
}

template<class Dist>
void TruncatedDist<Dist>::set_lbound(double lbound)
{
    set_bounds(lbound, ubound());
}

template<class Dist>
void TruncatedDist<Dist>::set_ubound(double ubound)
{
    set_bounds(lbound(), ubound);
}

template<class Dist>
double TruncatedDist<Dist>::cdf(double x) const
{
    return (this->Dist::cdf(x) - lbound_cdf) / bounds_cdf_delta;
}

template<class Dist>
double TruncatedDist<Dist>::icdf(double u) const
{
    return this->Dist::icdf(lbound_cdf + u*bounds_cdf_delta);
}

template<class Dist>
double TruncatedDist<Dist>::pdf(double x) const
{
    return this->Dist::pdf(x) / bounds_cdf_delta;
}

template<class Dist>
double TruncatedDist<Dist>::llh(double x) const
{
    return this->Dist::llh(x) + llh_truncation_const;
}

template<class Dist>
template<class RngT>
double TruncatedDist<Dist>::sample(RngT &rng) const
{
    if(!_truncated) return Dist::sample(rng);
    //If truncated, the iCDF method is the most generally applicable and efficient.
    //One nice property is we only need to draw a single RNG vs a rejection strategy
    std::uniform_real_distribution<double> uniform;
    return icdf(uniform(rng));
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_TRUNCATEDDIST_H */
