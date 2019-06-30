/** @file TruncatedDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief TruncatedDist declaration and templated methods definitions
 * 
 */
#ifndef PRIOR_HESSIAN_TRUNCATEDDIST_H
#define PRIOR_HESSIAN_TRUNCATEDDIST_H

#include <cmath>

#include "PriorHessian/Meta.h"
#include "PriorHessian/PriorHessianError.h"
#include "PriorHessian/BoundsAdaptedDist.h"

namespace prior_hessian {

/** @brief 
 * 
 */
template<class Dist>
class TruncatedDist : public Dist
{
public:
    static constexpr IdxT num_params() { return Dist::num_params(); } 
    static constexpr const double min_bounds_pdf_integral = 1.0e-8; /** minimum allowable delta in cdf for a valid truncation*/
    static double global_lbound() { return Dist::lbound(); }
    static double global_ubound() { return Dist::ubound(); }
    
    TruncatedDist(): TruncatedDist(Dist{}) { }
    TruncatedDist(double lbound, double ubound) : TruncatedDist(Dist{}, lbound, ubound) { }

    template<typename=meta::EnableIfNotIsSelfT<Dist,TruncatedDist>>
    TruncatedDist(const Dist &dist) : TruncatedDist(dist, Dist::lbound(), Dist::ubound()) { }
    
    template<typename=meta::EnableIfNotIsSelfT<Dist,TruncatedDist>>
    TruncatedDist(Dist &&dist) : TruncatedDist(std::move(dist), Dist::lbound(), Dist::ubound()) { }
    
    TruncatedDist(const Dist &dist, double lbound, double ubound) 
        : Dist(dist)
    { set_bounds(lbound,ubound); }

    TruncatedDist(Dist &&dist, double lbound, double ubound)
        : Dist(std::move(dist))
    { set_bounds(lbound,ubound); }

    double lbound() const { return _truncated_lbound; }
    double ubound() const { return _truncated_ubound; }
    bool truncated() const { return _truncated; }
    bool operator==(const TruncatedDist<Dist> &o) const 
    { 
        return _truncated_lbound==o._truncated_lbound && _truncated_ubound==o._truncated_ubound && 
                static_cast<const Dist&>(*this).operator==(static_cast<const Dist&>(o)); 
    }

    bool operator!=(const TruncatedDist<Dist> &o) const { return !this->operator==(o);}
    
    void set_bounds(double lbound, double ubound);    
    void set_lbound(double lbound);    
    void set_ubound(double ubound);    

    double mean() const { throw NotImplementedError("Mean is not implemented for truncated distributions. No general-purpose efficient algorithm."); }
    double median() const {return Dist::icdf(lbound_cdf+bounds_pdf_integral*.5); }
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
    double bounds_pdf_integral; // (cdf(_ubound) - cdf(_lbound))
    double llh_truncation_const;// -log(bounds_pdf_integral)   
};

template<class Dist>
void TruncatedDist<Dist>::set_bounds(double lbound, double ubound)
{
    if( !(lbound >= global_lbound()) ) {   //This form of comparison handles NaNs 
        std::ostringstream msg;
        msg<<"set_bounds: Invalid lbound:"<<lbound<<" with regard to global_lbound:"<<global_lbound();
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
    bool truncated = (lbound>global_lbound() || ubound<global_ubound());
    if(truncated) {
        lbound_cdf = (lbound==global_lbound()) ? 0 : Dist::cdf(lbound);
        double ubound_cdf = (ubound==global_ubound()) ? 1 : Dist::cdf(ubound);
        bounds_pdf_integral = ubound_cdf - lbound_cdf;
        if(bounds_pdf_integral<min_bounds_pdf_integral) {            
            std::ostringstream msg;
            msg<<"TruncatedDist::set_bounds: bounds:["<<lbound<<","<<ubound<<"] with cdf:["<<lbound_cdf<<","<<ubound_cdf
               <<"] have delta: "<<bounds_pdf_integral<<" < min_delta = "<<min_bounds_pdf_integral
               <<".  Bounds cover too small a portion of the domain for accuarate truncation.";
            throw ParameterValueError(msg.str());
        }
        llh_truncation_const = -log(bounds_pdf_integral);
    } else {
        lbound_cdf = 0;
        bounds_pdf_integral = 1;
        llh_truncation_const = 0;
    }
    _truncated = truncated;
    _truncated_lbound = lbound;
    _truncated_ubound = ubound;
}

template<class Dist>
void TruncatedDist<Dist>::set_lbound(double new_lbound)
{
    set_bounds(new_lbound, ubound());
}

template<class Dist>
void TruncatedDist<Dist>::set_ubound(double new_ubound)
{
    set_bounds(lbound(), new_ubound);
}

template<class Dist>
double TruncatedDist<Dist>::cdf(double x) const
{
    return (this->Dist::cdf(x) - lbound_cdf) / bounds_pdf_integral;
}

template<class Dist>
double TruncatedDist<Dist>::icdf(double u) const
{
    return this->Dist::icdf(lbound_cdf + u*bounds_pdf_integral);
}

template<class Dist>
double TruncatedDist<Dist>::pdf(double x) const
{
    return this->Dist::pdf(x) / bounds_pdf_integral;
}

template<class Dist>
double TruncatedDist<Dist>::llh(double x) const
{
    if(!std::isfinite(x))        throw ParameterValueError("GOO1");
    if(!std::isfinite(llh_truncation_const))        throw ParameterValueError("GOO2");
    if(!std::isfinite(this->Dist::llh(x))){
        std::cout<<"x: "<<x<<" llh: "<<this->Dist::llh(x)<<" params:"<<this->params().t()<<std::endl;
//         throw ParameterValueError("GOO3");
    }
    return this->Dist::llh(x) + llh_truncation_const;
}

template<class Dist>
template<class RngT>
double TruncatedDist<Dist>::sample(RngT &rng) const
{
    if(!truncated()) return Dist::sample(rng);
    //If truncated, the iCDF method is the most generally applicable and efficient.
    //One nice property is we only need to draw a single RNG vs a rejection strategy
    std::uniform_real_distribution<double> uniform;
    return icdf(uniform(rng));
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_TRUNCATEDDIST_H */
