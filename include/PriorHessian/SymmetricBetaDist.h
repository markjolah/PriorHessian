/** @file SymmetricBetaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief SymmetricBetaDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_SYMMETRICBETADIST_H
#define PRIOR_HESSIAN_SYMMETRICBETADIST_H

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
        /* Static constant member data */
    static const StringVecT param_names; //Cannonical names for parameters
    static const VecT param_lbound; //Lower bound on valid parameter values 
    static const VecT param_ubound; //Upper bound on valid parameter values
    /* Static member functions */
    static constexpr IdxT num_params() { return 1; }
    static bool check_params(double beta); /* Check parameters are valid (in bounds) */    
    static bool check_params(VecT &params);    /* Check a vector of parameters is valid (in bounds) */    
    template<class IterT>
    static bool check_params_iter(IterT &params);    /* Check a vector of parameters is valid (in bounds) */    
    
    SymmetricBetaDist(double beta=1.0);
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {_beta}; }
    void set_params(const VecT &p) { _beta = checked_beta(p[0]); }
    bool operator==(const SymmetricBetaDist &o) const { return _beta == o._beta; }
    bool operator!=(const SymmetricBetaDist &o) const { return !this->operator==(o);}

    double beta() const { return _beta; }
    void set_beta(double val) { _beta = checked_beta(val); }

    double mean() const { return 1/2; }
    double median() const { return icdf(0.5); }
    
    double cdf(double x) const;
    double icdf(double u) const;
    double pdf(double x) const;
    double llh(double x) const { return rllh(x) + llh_const; }
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;
    
    template<class RngT>
    double sample(RngT &rng) const;
private:
    using RngDistT = boost::math::beta_distribution<double>;//Used for RNG

    static double checked_beta(double val);
   
    double _beta; //distribution mean
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded normal dist uses the ScaledDist adaptor */
using ScaledSymmetricBetaDist = ScaledDist<SymmetricBetaDist>;

inline
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


inline
bool SymmetricBetaDist::check_params(double param0)
{
    return std::isfinite(param0) && param0>0;     
}

inline
bool SymmetricBetaDist::check_params(VecT &params)
{ 
    return std::isfinite(params[0]) && params[0]>0;     
}

template<class IterT>
bool SymmetricBetaDist::check_params_iter(IterT &p)
{ 
    return check_params(*p++);
}

inline
double SymmetricBetaDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _beta;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

inline
void SymmetricBetaDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_beta(val);
            return;
        default:
            return; //Don't handle indexing errors.
    }
}

inline
double SymmetricBetaDist::rllh(double x) const
{
    return (_beta-1) * log(x*(1-x));
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

template<class RngT>
double SymmetricBetaDist::sample(RngT &rng) const
{
    std::uniform_real_distribution<double> uni;
    return icdf(uni(rng));
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_SYMMETRICBETADIST_H */
