/** @file UpperTruncatedDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief UpperTruncatedDist declaration and templated methods definitions
 * 
 */
#ifndef _PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H
#define _PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H

#include <cmath>

#include "PriorHessian/PriorHessianError.h"

namespace prior_hessian {

/** @brief 
 * 
 */
template<class Dist>
class UpperTruncatedDist : public Dist
{
public:
    UpperTruncatedDist();
    explicit UpperTruncatedDist(double ubound);
    explicit UpperTruncatedDist(const Dist &dist);
    explicit UpperTruncatedDist(Dist &&dist);
    UpperTruncatedDist(const Dist &dist, double ubound);
    UpperTruncatedDist(Dist &&dist, double ubound);
    double ubound() const;
    double global_ubound() const;
    bool truncated() const;
    
    void set_bounds(double lbound, double ubound);    
    void set_ubound(double ubound);    

    double cdf(double x) const;
    double pdf(double x) const;
    double icdf(double u) const;
    double llh(double x) const;

    template<class RngT>
    double sample(RngT &rng) const;
protected:
    double _truncated_ubound;
    bool _truncated = false;

    double ubound_cdf; // cdf(_truncated_ubound)  [The lbound remains in place and the cdf at the lbound is 0 so this is equivalent to bounds_cdf_delta in TrunctedDist]
    double llh_truncation_const;// -log(ubounds_cdf)   
};

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist() 
    : UpperTruncatedDist(Dist{})
{ }

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist(double ubound) 
    : UpperTruncatedDist(Dist{}, ubound)
{ }

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist(const Dist &dist) 
    : UpperTruncatedDist(dist, dist.ubound())
{ }

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist(Dist &&dist)
    : UpperTruncatedDist(std::move(dist), dist.ubound())
{ }

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist(const Dist &dist, double ubound)
    : Dist(dist)
{ set_ubound(ubound); }

template<class Dist>
UpperTruncatedDist<Dist>::UpperTruncatedDist(Dist &&dist, double ubound)
    : Dist(std::move(dist))
{ set_ubound(ubound); }

template<class Dist>
double UpperTruncatedDist<Dist>::ubound() const
{ return _truncated_ubound; }

template<class Dist>
double UpperTruncatedDist<Dist>::global_ubound() const
{ return Dist::ubound(); }

template<class Dist>
bool UpperTruncatedDist<Dist>::truncated() const
{ return _truncated; }

template<class Dist>
void UpperTruncatedDist<Dist>::set_bounds(double lbound, double ubound)
{
    this->set_lbound(lbound);
    set_ubound(ubound);    
}

template<class Dist>
void UpperTruncatedDist<Dist>::set_ubound(double ubound)
{
    if( !(ubound >= global_ubound()) ) {   //This form of comparison handles NaNs
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<ubound<<" with regard to global_ubound:"<<global_ubound();
        throw ParameterValueError(msg.str());
    }
    if( !(this->lbound() < ubound) ) {
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<ubound<<" <= lbound: "<<this->lbound();
        throw ParameterValueError(msg.str());
    }
    _truncated =  ubound < global_ubound();
    if(_truncated) {
        ubound_cdf = Dist::cdf(ubound);
        llh_truncation_const = -log(ubound_cdf);
    } else {
        ubound_cdf = 1;
        llh_truncation_const = 0;
    }
    _truncated_ubound = ubound;
}

template<class Dist>
double UpperTruncatedDist<Dist>::cdf(double x) const
{
    return this->Dist::cdf(x) / ubound_cdf;
}

template<class Dist>
double UpperTruncatedDist<Dist>::icdf(double u) const
{
    return this->Dist::icdf(u*ubound_cdf);
}

template<class Dist>
double UpperTruncatedDist<Dist>::pdf(double x) const
{
    return this->Dist::pdf(x) / ubound_cdf;
}

template<class Dist>
double UpperTruncatedDist<Dist>::llh(double x) const
{
    return this->Dist::llh(x) + llh_truncation_const;
}

template<class Dist>
template<class RngT>
double UpperTruncatedDist<Dist>::sample(RngT &rng) const
{
    if(!_truncated) return Dist::sample(rng);
    //If truncated, the iCDF method is the most generally applicable and efficient.
    //One nice property is we only need to draw a single RNG vs a rejection strategy
    std::uniform_real_distribution<double> uniform;
    return icdf(uniform(rng));
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H */
