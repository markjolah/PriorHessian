/** @file NormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief NormalDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_NORMALDIST_H
#define _PRIOR_HESSIAN_NORMALDIST_H

#include <boost/math/special_functions/erf.hpp>

#include "TruncatingDist.h"

namespace prior_hessian {

/** @brief Normal distribution with truncation
 * 
 */
class NormalDist : public InfiniteDist<NormalDist>
{

public:
    NormalDist(double mean, double sigma, std::string var_name);
    NormalDist(double mean, double sigma, std::string var_name, StringVecT&& param_desc);
    NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name);
    NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();
    static StringVecT make_default_param_desc(std::string var_name);
    
    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params(IterT& p);   
        
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;

    double compute_llh_const() const;
    double unbounded_cdf(double x) const;
    double unbounded_icdf(double u) const;
    double unbounded_pdf(double x) const;

protected:
    constexpr static double pi=3.141592653589793;
    constexpr static double sqrt2 =  ::sqrt(2);
    constexpr static double sqrt2pi = ::sqrt(2*pi);
    constexpr static double log2pi = ::log(2*pi);
    
    double mean; //distribution mean
    double sigma; //distribution shape
    
    static void check_params(double mean_val, double sigma_val);
};

 
inline
NormalDist::NormalDist(double mean, double sigma, std::string var_name) :
    NormalDist(mean,sigma,-INFINITY,INFINITY,var_name,make_default_param_desc(var_name))
{ }

inline
NormalDist::NormalDist(double mean, double sigma, std::string var_name, StringVecT&& param_desc) :
    NormalDist(mean,sigma,-INFINITY,INFINITY,var_name,std::move(param_desc))
{ }

inline
NormalDist::NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name) :
    NormalDist(mean,sigma,lbound,ubound,var_name,make_default_param_desc(var_name))
{ }

inline
NormalDist::NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name, StringVecT&& param_desc) :
        InfiniteDist<NormalDist>(lbound,ubound,var_name,std::move(param_desc)),
        mean(mean),
        sigma(sigma)
{
    this->set_bounds(lbound,ubound);
    this->llh_const = compute_llh_const();
}

constexpr
IdxT NormalDist::num_params()
{ 
    return 2; 
}

inline
StringVecT NormalDist::make_default_param_desc(std::string var_name)
{
    return {std::string("mean_") + var_name, std::string("sigma_") + var_name};
}

inline
double NormalDist::unbounded_cdf(double x) const
{
    return .5*(1+boost::math::erf((x-mean)/(sqrt2*sigma)));
}

inline
double NormalDist::unbounded_icdf(double u) const
{
    return mean+sigma*sqrt2*boost::math::erf_inv(2*u-1);
}

inline
double NormalDist::unbounded_pdf(double x) const
{
    double val = (x-mean)/sigma;
    return exp(-.5*val*val)/(sigma*sqrt2pi);
}

inline
double NormalDist::compute_llh_const() const
{
    return -log(sigma) -.5*log2pi;
}

inline
double NormalDist::rllh(double x) const
{
    double val = (x-mean)/sigma;
    return -.5*val*val;
}

inline
double NormalDist::grad(double x) const
{
    return (x-mean)/(sigma*sigma);
}

inline
double NormalDist::grad2(double x) const
{
    return 1./(sigma*sigma);
}

inline
void NormalDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double s2inv = 1./(sigma*sigma);
    g  += (x-mean)*s2inv;
    g2 += -s2inv;
}

template<class IterT>
void NormalDist::append_params(IterT& p) const 
{ 
    *p++ = mean;
    *p++ = sigma;
} 

template<class IterT>
void NormalDist::set_params(IterT& p) 
{ 
    double mean_val = *p++;
    double sigma_val = *p++;
    check_params(mean_val, sigma_val);
    mean = mean_val;
    sigma = sigma_val;
    llh_const = compute_llh_const();
}     
    
inline
void NormalDist::check_params(double mean_val, double sigma_val) 
{ 
    if(!std::isfinite(mean_val)) {
        std::ostringstream msg;
        msg<<"NormalDist::set_params: got bad mean value:"<<mean_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
    if(sigma_val<=0 || !std::isfinite(sigma_val)) {
        std::ostringstream msg;
        msg<<"GammaDist::set_params: got bad sigma value:"<<sigma_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
}

    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_NORMALDIST_H */
