/** @file TruncatedMultivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief TruncatedMultivariateDist declaration and templated methods definitions
 * 
 */
#ifndef PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H
#define PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H

#include <cmath>

#include "PriorHessian/Meta.h"
#include "PriorHessian/PriorHessianError.h"
#include "PriorHessian/BoundsAdaptedDist.h"

namespace prior_hessian {

/** @brief 
 * 
 */
template<class Dist>
class TruncatedMultivariateDist : public Dist
{
public:
    using typename Dist::NdimVecT;
    static constexpr const double min_bounds_cdf_delta = 1.0e-8; /** minimum allowabale delta in cdf for a valid truncation*/
    
    TruncatedMultivariateDist(): TruncatedMultivariateDist(Dist{}) { }
    
    template<class Vec>
    TruncatedMultivariateDist(Vec &&lbound, Vec &&ubound) 
        : TruncatedMultivariateDist(Dist{}, std::forward<Vec>(lbound), std::forward<Vec>(ubound)) { }

    template<typename=meta::EnableIfNotIsSelfT<Dist,TruncatedMultivariateDist>>
    TruncatedMultivariateDist(const Dist &dist) : TruncatedMultivariateDist(dist, dist.lbound(), dist.ubound()) { }
    
    template<typename=meta::EnableIfNotIsSelfT<Dist,TruncatedMultivariateDist>>
    TruncatedMultivariateDist(Dist &&dist) : TruncatedMultivariateDist(std::move(dist), dist.lbound(), dist.ubound()) { }
    
    template<class Vec>
    TruncatedMultivariateDist(const Dist &dist, Vec &&lbound, Vec &&ubound) 
        : Dist(dist)
    { set_bounds(std::forward<Vec>(lbound),std::forward<Vec>(ubound)); }

    template<class Vec>
    TruncatedMultivariateDist(Dist &&dist, Vec &&lbound, Vec &&ubound) 
        : Dist(std::move(dist))
    { set_bounds(std::forward<Vec>(lbound),std::forward<Vec>(ubound)); }

    const NdimVecT& lbound() const { return _truncated_lbound; }
    const NdimVecT& ubound() const { return _truncated_ubound; }
    const NdimVecT& global_lbound() const { return Dist::lbound(); }
    const NdimVecT& global_ubound() const { return Dist::ubound(); }
    bool truncated() const { return _truncated; }
    bool operator==(const TruncatedMultivariateDist<Dist> &o) const 
    { 
        return arma::all(_truncated_lbound==o._truncated_lbound) &&arma::all( _truncated_ubound==o._truncated_ubound) && 
                static_cast<const Dist&>(*this).operator==(static_cast<const Dist&>(o)); 
    }

    bool operator!=(const TruncatedMultivariateDist<Dist> &o) const { return !this->operator==(o);}
    
    template<class Vec, class Vec2>
    void set_bounds(const Vec &lbound, const Vec2 &ubound);    
    template<class Vec>
    void set_lbound(const Vec &lbound);    
    template<class Vec>
    void set_ubound(const Vec &ubound);    
    
    double mean() const { throw NotImplementedError("No universal mean formula for truncated multivariate distributions"); }
    
    template<class Vec>
    double cdf(const Vec& x) const;
    template<class Vec>
    double pdf(const Vec& x) const;
    template<class Vec>
    double llh(const Vec& x) const;

    template<class RngT>
    NdimVecT sample(RngT &rng) const;
protected:
    NdimVecT _truncated_lbound;
    NdimVecT _truncated_ubound;
    bool _truncated = false;

    double lbound_cdf; // cdf(_lbound)
    double bounds_cdf_delta; // (cdf(_ubound) - cdf(_lbound))
    double llh_truncation_const;// -log(bounds_cdf_delta)   
};

template<class Dist>
template<class Vec, class Vec2>
void TruncatedMultivariateDist<Dist>::set_bounds(const Vec &lbound, const Vec2 &ubound)
{
    if( !arma::all(lbound >= global_lbound()) ) {   //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid lbound:"<<lbound.t()<<" with regard to global_lbound:"<<global_lbound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all(ubound <= global_ubound()) ) {    //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid ubound:"<<ubound.t()<<" with regard to global_ubound:"<<global_ubound().t();
        throw ParameterValueError(msg.str());
    }
    if( !arma::all(lbound < ubound) ) { 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid bounds lbound:"<<lbound.t()<<" >= ubound:"<<ubound.t();
        throw ParameterValueError(msg.str());
    }
    bool truncated = arma::any(lbound > global_lbound()) || arma::any(ubound < global_ubound());
    if(truncated) {
        lbound_cdf = arma::all(lbound==global_lbound()) ? 0 : Dist::cdf(lbound);
        double ubound_cdf = arma::all(ubound==global_ubound()) ? 1 : Dist::cdf(ubound);
        bounds_cdf_delta = ubound_cdf - lbound_cdf;
        if(bounds_cdf_delta < min_bounds_cdf_delta) {            
            std::ostringstream msg;
            msg<<"TruncatedMultivariateDist::set_bounds: params: ["<<this->params().t()<<"]\n bounds:[ ["<<lbound.t()<<"], ["<<ubound.t()<<"] ] with cdf:["<<lbound_cdf<<","<<ubound_cdf
               <<"] have delta: "<<bounds_cdf_delta<<" < min_delta = "<<min_bounds_cdf_delta
               <<".  Bounds cover too small a portation of the domain for accuarate truncation.";
            throw ParameterValueError(msg.str());
        }
        llh_truncation_const = -log(bounds_cdf_delta);
    } else {
        lbound_cdf = 0;
        bounds_cdf_delta = 1;
        llh_truncation_const = 0;
    }
    _truncated = truncated;
    _truncated_lbound = lbound;
    _truncated_ubound = ubound;
}

template<class Dist>
template<class Vec>
void TruncatedMultivariateDist<Dist>::set_lbound(const Vec &new_lbound)
{
    set_bounds(new_lbound, ubound());
}

template<class Dist>
template<class Vec>
void TruncatedMultivariateDist<Dist>::set_ubound(const Vec &new_ubound)
{
    set_bounds(lbound(), new_ubound);
}

template<class Dist>
template<class Vec>
double TruncatedMultivariateDist<Dist>::cdf(const Vec &x) const
{
    return (this->Dist::cdf(x) - lbound_cdf) / bounds_cdf_delta;
}

template<class Dist>
template<class Vec>
double TruncatedMultivariateDist<Dist>::pdf(const Vec &x) const
{
    return this->Dist::pdf(x) / bounds_cdf_delta;
}

template<class Dist>
template<class Vec>
double TruncatedMultivariateDist<Dist>::llh(const Vec &x) const
{
    return this->Dist::llh(x) + llh_truncation_const;
}

template<class Dist>
template<class RngT>
typename TruncatedMultivariateDist<Dist>::NdimVecT 
TruncatedMultivariateDist<Dist>::sample(RngT &rng) const
{
    if(!truncated()) return Dist::sample(rng);
    //If truncated, rejection sampling is the only universal multidimensionsal distribution.
    const int MaxIter = 1000;
    NdimVecT s;
    for(int n=0;n<MaxIter; n++){
        s=Dist::sample(rng);
        if(this->in_bounds(s)) return s;
    }
    throw RuntimeSamplingError("Truncated distribution rejection sampling failure.  A more efficient method is required.");
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H */
