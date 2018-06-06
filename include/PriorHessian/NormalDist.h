/** @file NormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief NormalDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_NORMALDIST_H
#define _PRIOR_HESSIAN_NORMALDIST_H

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>

#include "PriorHessian/TruncatingDist.h"

namespace prior_hessian {

/** @brief Normal distribution with truncation
 * 
 */
class NormalDist : public InfiniteDist<NormalDist>
{
public:
    NormalDist();
    NormalDist(double mean, double sigma, std::string var_name);
    NormalDist(double mean, double sigma, std::string var_name, StringVecT&& param_desc);
    NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name);
    NormalDist(double mean, double sigma, double lbound, double ubound, std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();
    double get_param(int idx) const;
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;

protected:
    constexpr static double sqrt2 =  ::sqrt(2);
    constexpr static double sqrt2pi_inv = 1./::sqrt(2*boost::math::double_constants::pi);
    constexpr static double log2pi = ::log(2*boost::math::double_constants::pi);

    double compute_llh_const() const;
    double unbounded_cdf(double x) const;
    double unbounded_icdf(double u) const;
    double unbounded_pdf(double x) const;

    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params_iter(IterT& p);   
    
    double mean; //distribution mean
    double sigma; //distribution shape
    double sigma_inv;
    
    static StringVecT make_default_param_desc(std::string var_name);
    static void check_params(double mean_val, double sigma_val);

    friend UnivariateDist<NormalDist>;
    friend InfiniteDist<NormalDist>;
    friend TruncatingDist<NormalDist>;
    template<class RngT> friend class CompositeDist;
};

inline
NormalDist::NormalDist() :
    NormalDist(0,1,-INFINITY,INFINITY,"x",make_default_param_desc("x"))
{ }

 
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
        sigma(sigma),
        sigma_inv(1./sigma)
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
double NormalDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return mean;
        case 1:
            return sigma;
        default:
            std::ostringstream msg;
            msg<<"Bad parameter index: "<<idx<<" max:"<<num_params();
            throw IndexError(msg.str());
    }
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
    double val = (x-mean)*sigma_inv;
    return exp(-.5*val*val)*sigma_inv*sqrt2pi_inv;
}

inline
double NormalDist::compute_llh_const() const
{
    return -log(sigma) -.5*log2pi;
}

inline
double NormalDist::rllh(double x) const
{
    double val = (x-mean)*sigma_inv;
    return -.5*val*val;
}

inline
double NormalDist::grad(double x) const
{
    return -(x-mean)*(sigma_inv*sigma_inv);
}

inline
double NormalDist::grad2(double x) const
{
    return -sigma_inv*sigma_inv;
}

inline
void NormalDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double sigma_inv2 = sigma_inv*sigma_inv;
    g  += -(x-mean)*sigma_inv2;
    g2 += -sigma_inv2;
}

template<class IterT>
void NormalDist::append_params(IterT& p) const 
{ 
    *p++ = mean;
    *p++ = sigma;
} 

template<class IterT>
void NormalDist::set_params_iter(IterT& p) 
{ 
    double mean_val = *p++;
    double sigma_val = *p++;
    check_params(mean_val, sigma_val);
    mean = mean_val;
    sigma = sigma_val;
    sigma_inv = 1./sigma;
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
