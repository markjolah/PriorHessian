/** @file SymmetricBetaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief SymmetricBetaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_SYMMETRICBETADIST_H
#define _PRIOR_HESSIAN_SYMMETRICBETADIST_H

#include <cmath>
#include <random>

#include <boost/math/distributions/beta.hpp>

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/ScaledDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class SymmetricBetaDist : public UnivariateDist
{
public:
    static const StringVecT param_names;
    static constexpr IdxT num_params();
    SymmetricBetaDist(double beta=1.0);
double get_param(int idx) const;
    void set_param(int idx, double val);
    double beta() const;
    void set_beta(double val);
        
    double cdf(double x) const;
    double icdf(double u) const;
    double pdf(double x) const;
    double llh(double x) const;
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;
    
    template<class RngT>
    double sample(RngT &rng) const;
protected:
    using RngDistT = boost::math::beta_distribution<double>;//Used for RNG

    static double check_beta(double val);
   
    double _beta; //distribution mean
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded normal dist uses the ScaledDist adaptor */
using ScaledSymmetricBetaDist = ScaledDist<SymmetricBetaDist>;

ScaledSymmetricBetaDist make_scaled_symmetric_beta_dist(double beta, double lbound, double ubound)
{
    return {SymmetricBetaDist(beta),lbound,ubound};
}

namespace detail
{
    template<class Dist>
    class dist_adaptor_traits;
    
    template<>
    class dist_adaptor_traits<SymmetricBetaDist> {
    public:
        using bounds_adapted_dist = ScaledSymmetricBetaDist;
        
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    class dist_adaptor_traits<ScaledSymmetricBetaDist> {
    public:
        using bounds_adapted_dist = ScaledSymmetricBetaDist;
        
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace detail */

constexpr
IdxT SymmetricBetaDist::num_params()
{ return 1; }

inline
double SymmetricBetaDist::beta() const
{ return _beta; }

inline
void SymmetricBetaDist::set_beta(double val)
{ _beta = check_beta(val); }

inline
double SymmetricBetaDist::rllh(double x) const
{
    return (_beta-1) * log(x*(1-x));
}

inline
double SymmetricBetaDist::llh(double x) const
{
    return rllh(x) + llh_const;
}

inline
double SymmetricBetaDist::grad(double x) const
{
    return (_beta-1) * (1/x-1/(1-x));
}

inline
double SymmetricBetaDist::grad2(double x) const
{
    return (_beta-1) * (square(1/(1-x)) - square(1/x));
}

inline
void SymmetricBetaDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double v= 1/(1-x);
    double x_inv = 1/x;
    double bm1 = _beta-1;
    g  += bm1*(x_inv-v); //(beta-1)*(1/z-1/(1-z))
    g2 += bm1*(v*v-x_inv*x_inv);
}
       
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_SYMMETRICBETADIST_H */
