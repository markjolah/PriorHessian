/** @file NormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief NormalDist class declaration and templated methods
 * 
 */

#ifndef PRIOR_HESSIAN_NORMALDIST_H
#define PRIOR_HESSIAN_NORMALDIST_H

#include <cmath>
#include <random>

#include "PriorHessian/UnivariateDist.h"

namespace prior_hessian {

/** @brief Normal distribution with truncation
 * 
 */
class NormalDist : public UnivariateDist
{
public:
    /* Static member functions */
    static constexpr IdxT num_params() { return 2; }
    
    static const StringVecT& param_names();
    static const VecT& param_lbound();
    static const VecT& param_ubound();
    
    static bool check_params(double mu, double sigma); /* Check parameters are valid (in bounds) */    
    static bool check_params(const VecT &params);    /* Check a vector of parameters is valid (in bounds) */    

    /* Constructor */
    NormalDist(double mu=0.0, double sigma=1.0);
    NormalDist(const VecT &params);
    
    /* Member functions */
    double mu() const;
    double sigma() const;
    void set_mu(double val);
    void set_sigma(double val);
    bool operator==(const NormalDist &o) const { return mu() == o.mu() && sigma() == o.sigma(); }
    bool operator!=(const NormalDist &o) const { return !this->operator==(o);}
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {mu(), sigma()}; }
    void set_params(double mu, double sigma);
    void set_params(const VecT &p);
    
    double mean() const { return mu(); }
    double median() const { return mu(); }
    
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

     /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);   
    
    template<class IterT>
    void set_params_iter(IterT &params);
    
private:
    using RngDistT = std::normal_distribution<double>; //Used for RNG

    static const StringVecT _param_names; //Cannonical names for parameters
    static const VecT _param_lbound; //Lower bound on valid parameter values 
    static const VecT _param_ubound; //Upper bound on valid parameter values

    static double checked_mu(double val);
    static double checked_sigma(double val);
   
    double _mu; //distribution mu
    double _sigma_inv; //distribution shape
    double _sigma; //Keep actual sigma also to preserve exact replication of input sigma. 1./(1./sigma) != sigma in general.

    //Lazy computation of llh_const.  Most use-cases do not need it.
    mutable double llh_const;
    mutable bool llh_const_initialized;
    void initialize_llh_const() const;
    static double compute_llh_const(double sigma);
};

/* static methods */
inline
const StringVecT& NormalDist::param_names() 
{ return _param_names; }

inline
const VecT& NormalDist::param_lbound() 
{ return _param_lbound; }

inline
const VecT& NormalDist::param_ubound() 
{ return _param_ubound; }
 

/* non-static methods */
inline
double NormalDist::mu() const { return _mu; }

inline
double NormalDist::sigma() const { return _sigma; }

inline
void NormalDist::set_mu(double val) { _mu = checked_mu(val); }

inline
void NormalDist::set_params(double _mu, double _sigma)
{
    set_mu(_mu);
    set_sigma(_sigma);
}

inline
bool NormalDist::check_params(double mu, double sigma)
{
    return std::isfinite(mu) && std::isfinite(sigma) && sigma>0;     
}

inline
bool NormalDist::check_params(const VecT &params)
{ 
    return params.is_finite() && params(1)>0;
}

inline
double NormalDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return mu();
        case 1:
            return sigma();
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
void NormalDist::set_params(const VecT &p)
{ 
    set_mu(p(0));
    set_sigma(p(1));
}

inline
double NormalDist::pdf(double x) const
{
    return exp(-.5*square((x - _mu)*_sigma_inv))*constants::sqrt2pi_inv*_sigma_inv;
}

inline
double NormalDist::rllh(double x) const
{
    return -.5*square((x - _mu)*_sigma_inv);
}

inline
double NormalDist::grad(double x) const
{
    return -(x - _mu)*square(_sigma_inv);
}

inline
double NormalDist::grad2(double ) const
{
    return -square(_sigma_inv);
}

inline
void NormalDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double sigma_inv2 = square(_sigma_inv);
    g  += -(x - _mu)*sigma_inv2;
    g2 += -sigma_inv2;
}

template<class RngT>
double NormalDist::sample(RngT &rng) const
{
    RngDistT d(mu(),sigma());
    return d(rng);
}
 
 
/* Protected methods */
template<class IterT>
bool NormalDist::check_params_iter(IterT &params)
{
    double mu = *params++;
    double sigma = *params++;
    return check_params(mu,sigma);
}

template<class IterT>
void NormalDist::set_params_iter(IterT &params)
{
    double mu = *params++;
    double sigma = *params++;
    return set_params(mu,sigma);
}
 
 
} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_NORMALDIST_H */
