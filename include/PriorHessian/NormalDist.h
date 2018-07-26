/** @file NormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief NormalDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_NORMALDIST_H
#define _PRIOR_HESSIAN_NORMALDIST_H

#include <cmath>
#include <random>

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/TruncatedDist.h"

namespace prior_hessian {

/** @brief Normal distribution with truncation
 * 
 */
class NormalDist : public UnivariateDist
{
public:
    /* Static constant member data */
    static const StringVecT param_names; //Cannonical names for parameters
    static const VecT param_lbound; //Lower bound on valid parameter values 
    static const VecT param_ubound; //Upper bound on valid parameter values
    /* Static member functions */
    static constexpr IdxT num_params() { return 2; }
    static bool check_params(double mu, double sigma); /* Check parameters are valid (in bounds) */    
    static bool check_params(VecT &params);    /* Check a vector of parameters is valid (in bounds) */    
    template<class IterT>
    static bool check_params_iter(IterT &params);    /* Check a vector of parameters is valid (in bounds) */    
    
    NormalDist(double mu=0.0, double sigma=1.0);
    
    double mu() const { return _mu; }
    double sigma() const { return _sigma; }
    void set_mu(double val) { _mu = checked_mu(val); }
    void set_sigma(double val) { _sigma = checked_sigma(val); }
    bool operator==(const NormalDist &o) const { return _mu == o._mu && _sigma == o._sigma; }
    bool operator!=(const NormalDist &o) const { return !this->operator==(o);}
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {_mu, _sigma}; }
    void set_params(const VecT &p)
    { 
        _mu = checked_mu(p[0]);  
        _sigma = checked_sigma(p[1]); 
    }
    
    double mean() const { return _mu; }
    double median() const { return _mu; }
    
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

protected:
    using RngDistT = std::normal_distribution<double>; //Used for RNG

    static const double sqrt2;
    static const double sqrt2pi_inv;
    static const double log2pi;

    static double checked_mu(double val);
    static double checked_sigma(double val);
   
    double _mu; //distribution mu
    double _sigma; //distribution shape
    double sigma_inv; //Pre-compute this to eliminate divisions in many computations
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded normal dist uses the TruncatedDist adaptor */
using BoundedNormalDist = TruncatedDist<NormalDist>;

inline
BoundedNormalDist make_bounded_normal_dist(double mu, double sigma, double lbound, double ubound)
{
    return {NormalDist(mu, sigma),lbound,ubound};
}

namespace detail
{
    template<class Dist>
    class dist_adaptor_traits;
    
    template<>
    class dist_adaptor_traits<NormalDist> {
    public:
        using bounds_adapted_dist = BoundedNormalDist;
        
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    class dist_adaptor_traits<BoundedNormalDist> {
    public:
        using bounds_adapted_dist = BoundedNormalDist;
        
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace detail */

inline
bool NormalDist::check_params(double param0, double param1)
{
    return std::isfinite(param0) && std::isfinite(param1);     
}

inline
bool NormalDist::check_params(VecT &params)
{ 
    return params.is_finite();     
}

template<class IterT>
bool NormalDist::check_params_iter(IterT &params)
{
    double mu = *params++;
    double sigma = *params++;
    return check_params(mu,sigma);
}

inline
double NormalDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _mu;
        case 1:
            return _sigma;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

inline
void NormalDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_mu(val);
            return;
        case 1:
            set_sigma(val);
            return;
        default:
            return; //Don't handle indexing errors.
    }
}


inline
double NormalDist::pdf(double x) const
{
    double val = (x - _mu)*sigma_inv;
    return exp(-.5*val*val)*sigma_inv*sqrt2pi_inv;
}

inline
double NormalDist::compute_llh_const() const
{
    return -log(_sigma) -.5*log2pi;
}

inline
double NormalDist::rllh(double x) const
{
    double val = (x - _mu)*sigma_inv;
    return -.5*val*val;
}

inline
double NormalDist::grad(double x) const
{
    return -(x - _mu)*sigma_inv*sigma_inv;
}

inline
double NormalDist::grad2(double ) const
{
    return -sigma_inv*sigma_inv;
}

inline
void NormalDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double sigma_inv2 = sigma_inv*sigma_inv;
    g  += -(x - _mu)*sigma_inv2;
    g2 += -sigma_inv2;
}

template<class RngT>
double NormalDist::sample(RngT &rng) const
{
    RngDistT d(_mu,_sigma);
    return d(rng);
}
    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_NORMALDIST_H */
