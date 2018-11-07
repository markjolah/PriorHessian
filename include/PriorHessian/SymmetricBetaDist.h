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

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class SymmetricBetaDist : public UnivariateDist
{
    /* These paramter vectors are constant sized, but are handled by accessor functions
     * so as to work generically together with multivariate distributions.
     */
    static const StringVecT _param_names; //Cannonical names for parameters
    static const VecT _param_lbound; //Lower bound on valid parameter values 
    static const VecT _param_ubound; //Upper bound on valid parameter values
public:
    /* Static constant member data */
    static const StringVecT& param_names() { return _param_names; }
    static const VecT& param_lbound() { return _param_lbound; }
    static const VecT& param_ubound() { return _param_ubound; }

    /* Static member functions */
    static constexpr IdxT num_params() { return 1; }
    static bool check_params(double beta); /* Check parameters are valid (in bounds) */    
    static bool check_params(VecT &params);    /* Check a vector of parameters is valid (in bounds) */    
    
    SymmetricBetaDist(double beta=1.0);

    double beta() const { return _beta; }
    void set_beta(double val);

    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {beta()}; }
    void set_params(double beta) { set_beta(beta); }
    void set_params(const VecT &p) { set_beta(p[0]); }
    bool operator==(const SymmetricBetaDist &o) const { return beta() == o.beta(); }
    bool operator!=(const SymmetricBetaDist &o) const { return !this->operator==(o);}

    double mean() const { return 1/2; }
    double median() const { return icdf(0.5); }
    
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
    using RngDistT = boost::math::beta_distribution<double>;//Used for RNG

    static double checked_beta(double val);
   
    double _beta; //distribution mean
    
    //Lazy computation of llh_const.  Most use-cases do not need it.
    mutable double llh_const;
    mutable bool llh_const_initialized;
    void initialize_llh_const() const;
    static double compute_llh_const(double beta);
};

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
    double v = 1/(1-x);
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

/* Protected methods */
template<class IterT>
bool SymmetricBetaDist::check_params_iter(IterT &p)
{ 
    return check_params(*p++);
}

template<class IterT>
void SymmetricBetaDist::set_params_iter(IterT &p)
{ 
    set_params(*p++);
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_SYMMETRICBETADIST_H */
