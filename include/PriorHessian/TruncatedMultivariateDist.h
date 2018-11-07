/** @file TruncatedMultivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief TruncatedMultivariateDist declaration and templated methods definitions
 * 
 */
#ifndef PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H
#define PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H

#include <cmath>
#include <bitset>

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
    static constexpr const double min_bounds_pdf_integral = 1.0e-8; /** minimum allowabale integral of pdf for a valid truncation*/
    
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
    template<class Vec>
    bool in_bounds(const Vec &u) const{ return arma::all(lbound()<=u) && arma::all(u<=ubound()); }
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

    double compute_truncated_pdf_integral(const NdimVecT &lbound, const NdimVecT &ubound, double lbound_cdf) const;
    double lbound_cdf; // cdf(lbound())
    double bounds_pdf_integral; // integral of pdf over valid bounded polytope
    double llh_truncation_const;// -log(bounds_pdf_integral)   
};

template<class Dist>
double TruncatedMultivariateDist<Dist>::compute_truncated_pdf_integral(const NdimVecT &lbound, const NdimVecT &ubound, double lbound_cdf) const
{
    const IdxT N = Dist::num_dim();
    if(lbound_cdf==0 && arma::all(lbound==-INFINITY)) return this->Dist::cdf(ubound);
    double pdf_integral = (N%2==0) ? lbound_cdf : -lbound_cdf; //account for the lbound() vertex here.

    //n iterates through all integers less than 2^N. we use the binary repr of N to choose ubound or lbound
    // 0 bit = use ubound(k)
    // 1 bit = use lbound(k)
    // all 1's is skipped as we account for that in initial pdf_integral value
    for(IdxT n=0; n<pow(2,N)-1; n++) {
        IdxT b=n; //b is used as a binary repr.
        IdxT k=0; //k is index we are considering
        IdxT flips=0;
        NdimVecT v = ubound;
//         std::cout<<"N:"<<N<<" n:"<<n<<" ["<<std::bitset<8>(n)<<"]\n";
        while(b && k<N) {
//             std::cout<<"k:"<<k<<" b:"<<std::bitset<8>(b)<<"\n";
            if(b&0x1) {
                v(k) = lbound(k);
                flips++;
            }
            k++;
            b>>=1;
        }
//         std::cout<<"Ubound:"<<ubound.t();
//         std::cout<<"     v:"<<v.t();
//         std::cout<<"Lbound:"<<lbound.t();
        double cdf_val = arma::any(v==-INFINITY) ? 0 : this->Dist::cdf(v);
//         std::cout<<"   cdf:"<<cdf_val<<"\n";
//         std::cout<<" flips:"<<flips<<"\n";
        if(flips%2==1) cdf_val = -cdf_val; //odd number of flips
        pdf_integral += cdf_val;
    }
    return pdf_integral;
}
        
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
        lbound_cdf = arma::any(lbound==-INFINITY) ? 0 : this->Dist::cdf(lbound);
        bounds_pdf_integral = compute_truncated_pdf_integral(lbound,ubound,lbound_cdf);
        if(!(bounds_pdf_integral > min_bounds_pdf_integral)) {            
            std::ostringstream msg;
            msg<<"TruncatedMultivariateDist::set_bounds: params: ["<<this->params().t()<<"]\n bounds:[ ["<<lbound.t()<<"], ["<<ubound.t()<<"] ] with cdf:["<<lbound_cdf<<","<<this->Dist::cdf(ubound)
               <<"] have pdf integral: "<<bounds_pdf_integral<<" < min_delta = "<<min_bounds_pdf_integral
               <<".  Bounds cover too small a portion of the domain for accuarate truncation.";
            throw ParameterValueError(msg.str());
        }
        llh_truncation_const = -log(bounds_pdf_integral);
//         std::cout<<"trunc?:"<<truncated<<" lbound:"<<lbound.t()<<" ubound:"<<ubound.t()<<" lboundcdf:"<<this->Dist::cdf(lbound)<<" internal lbound_cdf:"<<lbound_cdf<<" uboundcdf:"<<this->Dist::cdf(ubound)
//         <<" bounds_pdf_integral:"<<bounds_pdf_integral<<" llh_trunc:"<<llh_truncation_const<<"\n";
    } else {
        bounds_pdf_integral = 1;
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
    if(!truncated()) return this->Dist::cdf(x);
    return compute_truncated_pdf_integral(lbound(),ubound(),lbound_cdf);
}

template<class Dist>
template<class Vec>
double TruncatedMultivariateDist<Dist>::pdf(const Vec &x) const
{
    return this->Dist::pdf(x) / bounds_pdf_integral;
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
    const int MaxIter = 100000;
    NdimVecT s;
    for(int n=0;n<MaxIter; n++){
        s=Dist::sample(rng);
        if(this->in_bounds(s)) {
//             std::cout<<"Nsamples: "<<n<<"\n";
            return s;
        }
    }
    throw RuntimeSamplingError("Truncated distribution rejection sampling failure.  A more efficient method is required.");
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TRUNCATEDMULTIVARIATEDIST_H */
