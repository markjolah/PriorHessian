/** @file UpperTruncatedDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief UpperTruncatedDist declaration and templated methods definitions
 * 
 */
#ifndef PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H
#define PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H

#include <cmath>

#include "PriorHessian/Meta.h"
#include "PriorHessian/PriorHessianError.h"
#include "PriorHessian/BoundsAdaptedDist.h"

namespace prior_hessian {

/** @brief 
 * 
 */
template<class Dist>
class UpperTruncatedDist : public Dist
{
public:
    UpperTruncatedDist() : UpperTruncatedDist(Dist{}) { }
    explicit UpperTruncatedDist(double ubound) : UpperTruncatedDist(Dist{}, ubound) { }
    
    template<typename=meta::EnableIfNotIsSelfT<Dist,UpperTruncatedDist>>
    UpperTruncatedDist(const Dist &dist) : UpperTruncatedDist(dist, dist.ubound()) { }
    
    template<typename=meta::EnableIfNotIsSelfT<Dist,UpperTruncatedDist>>
    UpperTruncatedDist(Dist &&dist) : UpperTruncatedDist(std::move(dist), dist.ubound()) { }
    
    UpperTruncatedDist(const Dist &dist, double ubound) : Dist(dist) { set_ubound(ubound); }
    UpperTruncatedDist(Dist &&dist, double ubound) : Dist(std::move(dist)) { set_ubound(ubound); }

    double ubound() const { return _truncated_ubound; }
    double global_ubound() const { return Dist::ubound(); }
    bool truncated() const { return _truncated; }
    bool operator==(const UpperTruncatedDist<Dist> &o) const 
    { 
        return  _truncated_ubound==o._truncated_ubound && 
                static_cast<const Dist&>(*this).operator==(static_cast<const Dist&>(o)); 
    }

    bool operator!=(const UpperTruncatedDist<Dist> &o) const { return !this->operator==(o);}
     
    void set_bounds(double lbound, double ubound);    
    void set_lbound(double ubound);    
    void set_ubound(double ubound);    

    double mean() const { throw NotImplementedError("Mean is not implemented for truncated distributions. No general-purpose efficient algorithm."); }
    double median() const {return Dist::icdf((Dist::cdf(this->lbound())+ubound_cdf)*.5); }
    
    double cdf(double x) const;
    double pdf(double x) const;
    double icdf(double u) const;
    double llh(double x) const;

    template<class RngT>
    double sample(RngT &rng) const;
private:
    double _truncated_ubound;
    bool _truncated = false;

    double ubound_cdf; // cdf(_truncated_ubound)  [The lbound remains in place and the cdf at the lbound is 0 so this is equivalent to bounds_cdf_delta in TrunctedDist]
    double llh_truncation_const;// -log(ubounds_cdf)   

    void set_ubound_impl(double ubound);
};

template<class Dist>
void UpperTruncatedDist<Dist>::set_bounds(double lbound, double ubound)
{
    if(!Dist::check_lbound(lbound)) {
        std::ostringstream msg;
        msg<<"set_bounds: Invalid lbound:"<<lbound;
        throw ParameterValueError(msg.str());
    }
    if( !(ubound <= global_ubound()) ) {    //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<ubound<<" with regard to global_ubound:"<<global_ubound();
        throw ParameterValueError(msg.str());
    }
    if( !(lbound < ubound) ) { 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid bounds lbound:"<<lbound<<" >= ubound:"<<ubound;
        throw ParameterValueError(msg.str());
    }
    static_cast<Dist*>(this)->set_lbound(lbound);
    set_ubound_impl(ubound);  
}

template<class Dist>
void UpperTruncatedDist<Dist>::set_lbound(double lbound)
{
    if(!Dist::check_lbound(lbound)) {
        std::ostringstream msg;
        msg<<"set_lbound: Invalid lbound:"<<lbound;
        throw ParameterValueError(msg.str());
    }
    if( !(lbound < this->ubound()) ) {
        std::ostringstream msg;
        msg<<"set_lbound: ubound:"<<this->ubound()<<" <= lbound: "<<lbound;
        throw ParameterValueError(msg.str());
    }
    static_cast<Dist*>(this)->set_lbound(lbound);
}

template<class Dist>
void UpperTruncatedDist<Dist>::set_ubound(double ubound)
{
    if( !(ubound <= global_ubound()) ) {   //This form of comparison handles NaNs
        std::ostringstream msg;
        msg<<"set_ubound: Invalid ubound:"<<ubound<<" with regard to global_ubound:"<<global_ubound();
        throw ParameterValueError(msg.str());
    }
    if( !(this->lbound() < ubound) ) {
        std::ostringstream msg;
        msg<<"set_ubound: Invalid ubound:"<<ubound<<" <= lbound: "<<this->lbound();
        throw ParameterValueError(msg.str());
    }
    set_ubound_impl(ubound);
}

template<class Dist>
void UpperTruncatedDist<Dist>::set_ubound_impl(double ubound)
{
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

#endif /* PRIOR_HESSIAN_UPPERTRUNCATEDDIST_H */
