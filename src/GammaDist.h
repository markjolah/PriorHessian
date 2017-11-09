/** @file GammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief GammaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_GAMMADIST_H
#define _PRIOR_HESSIAN_GAMMADIST_H
#include <trng/gamma_dist.hpp>
#include "UnivariateDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class GammaDist : public UnivariateDist<GammaDist>
{

public:
    GammaDist(double mean, double kappa, std::string var_name);
    GammaDist(double mean, double kappa, std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();
    static StringVecT make_default_param_desc(std::string var_name);
    static double compute_llh_const(double mean, double kappa);

    template<class IterT> void insert_params(IterT& p) const;
    template<class IterT> void set_params(IterT& p);   
    
    double cdf(double x) const;
    double pdf(double x) const;
    double llh(double x) const;
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;

    template<class RngT> double sample(RngT &rng);
protected:
    using RNGDistT = trng::gamma_dist<double>;
    double mean; //distribution mean
    double kappa; //distribution shape
    RNGDistT dist;
    double llh_const; //Constant term of log-likelihood
};

inline
GammaDist::GammaDist(double mean, double kappa, std::string var_name) :
        UnivariateDist<GammaDist>(0,INFINITY,var_name,make_default_param_desc(var_name)),
        mean(mean),
        kappa(kappa),
        dist(kappa,mean/kappa),
        llh_const(compute_llh_const(mean,kappa))
{
}


inline
GammaDist::GammaDist(double mean, double kappa, std::string var_name, StringVecT&& param_desc) :
        UnivariateDist<GammaDist>(0,INFINITY,var_name,std::move(param_desc)),
        mean(mean),
        kappa(kappa),
        dist(kappa,mean/kappa),
        llh_const(compute_llh_const(mean,kappa))
{
}

constexpr
IdxT GammaDist::num_params()
{ 
    return 2; 
}

inline
StringVecT GammaDist::make_default_param_desc(std::string var_name)
{
    return {std::string("mean_") + var_name, std::string("kappa_") + var_name};
}

inline
double GammaDist::compute_llh_const(double mean, double kappa)
{
    return kappa*(log(kappa/mean))-lgamma(kappa);
}

inline
double GammaDist::cdf(double x) const
{
    return dist.cdf(x);
}

inline
double GammaDist::pdf(double x) const
{
    return dist.pdf(x);
}

inline
double GammaDist::llh(double x) const
{
    return rllh(x) + llh_const;
}

inline
double GammaDist::rllh(double x) const
{
    return (kappa-1)*log(x) - kappa*x/mean;
}

inline
double GammaDist::grad(double x) const
{
    return (kappa-1)/x - kappa/mean;
}

inline
double GammaDist::grad2(double x) const
{
    return -(kappa-1)/(x*x);
}

inline
void GammaDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double km1 = kappa-1;
    g  += km1/x - kappa/mean;
    g2 += -km1/(x*x);
}

/* Templated method definitions */

template<class IterT>
void GammaDist::insert_params(IterT& p) const 
{ 
    *p++ = mean;
    *p++ = kappa;
} 

template<class IterT>
void GammaDist::set_params(IterT& p) 
{ 
    mean = *p++;
    kappa = *p++;
    dist = RNGDistT(kappa,mean/kappa);
    llh_const = compute_llh_const(mean,kappa);
}     

template<class RngT> 
double GammaDist::sample(RngT &rng) 
{ 
    return dist(rng);
}
    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_GAMMADIST_H */
