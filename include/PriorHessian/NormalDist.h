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
    static const StringVecT param_names;
    static constexpr IdxT num_params();
    
    NormalDist(double mu=0.0, double sigma=1.0);
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    double mu() const;
    double sigma() const;
    void set_mu(double val);
    void set_sigma(double val);
        
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
    using RngDistT = std::normal_distribution<double>; //Used for RNG

    static const double sqrt2;
    static const double sqrt2pi_inv;
    static const double log2pi;

    static double check_mu(double val);
    static double check_sigma(double val);
   
    double _mu; //distribution mu
    double _sigma; //distribution shape
    double sigma_inv; //Pre-compute this to eliminate divisions in many computations
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded normal dist uses the TruncatedDist adaptor */
using BoundedNormalDist = TruncatedDist<NormalDist>;

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

constexpr
IdxT NormalDist::num_params()
{ return 2; }

inline
double NormalDist::mu() const
{ return _mu; }

inline
double NormalDist::sigma() const
{ return _sigma; }

inline
void NormalDist::set_mu(double val)
{ _mu = check_mu(val); }

inline
void NormalDist::set_sigma(double val)
{ _sigma = check_sigma(val); }

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
double NormalDist::llh(double x) const
{
    return rllh(x) + llh_const;
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
