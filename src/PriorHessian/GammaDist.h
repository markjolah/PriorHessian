/** @file GammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief GammaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_GAMMADIST_H
#define _PRIOR_HESSIAN_GAMMADIST_H

#include <boost/math/distributions/gamma.hpp>
#include "TruncatingDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 * We parameterize by mean= (kappa*theta) and kappa instead of the normal parameters: 
 * (theta,kappa).  The parameterization by mean is more useful for the context of a prior
 * because the expert users are expected to tune the prior distribution and a mean an shape is more 
 * inuititive then a shape and scale where the scale does not correspond directly to a physical scale of
 * interest in the system or experiment. 
 */
class GammaDist : public SemiInfiniteDist<GammaDist>
{

public:
    GammaDist(double mean, double kappa, std::string var_name);
    GammaDist(double mean, double kappa, std::string var_name, StringVecT&& param_desc);
    GammaDist(double mean, double kappa, double lbound, double ubound, std::string var_name);
    GammaDist(double mean, double kappa, double lbound, double ubound, std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();
    static StringVecT make_default_param_desc(std::string var_name);
    
    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params(IterT& p);   
        
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;

    friend class TruncatingDist<GammaDist>;

    double compute_llh_const() const;
    double unbounded_cdf(double x) const;
    double unbounded_icdf(double u) const;
    double unbounded_pdf(double x) const;

protected:
    using DistT = boost::math::gamma_distribution<double>;
    double mean; //distribution mean
    double kappa; //distribution shape
    DistT dist; //Boost distribution for use in computations of cdf,pdf,icdf    
    
    static void check_params(double mean_val, double kappa_val);
};

inline
GammaDist::GammaDist(double mean, double kappa, std::string var_name) :
    GammaDist(mean, kappa, 0, INFINITY, var_name, make_default_param_desc(var_name))
{ }

inline
GammaDist::GammaDist(double mean, double kappa, std::string var_name, StringVecT&& param_desc) :
    GammaDist(mean, kappa, 0, INFINITY, var_name, std::move(param_desc))
{ }

inline
GammaDist::GammaDist(double mean, double kappa, double lbound, double ubound, std::string var_name) :
    GammaDist(mean, kappa,  lbound, ubound, var_name, make_default_param_desc(var_name))
{ }

inline
GammaDist::GammaDist(double mean, double kappa, double lbound, double ubound, std::string var_name, StringVecT&& param_desc) :
        SemiInfiniteDist<GammaDist>(lbound,ubound,var_name,std::move(param_desc)),
        mean(mean),
        kappa(kappa),
        dist(kappa,mean/kappa)
{   
    this->set_bounds(lbound,ubound);
    this->llh_const = compute_llh_const();
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
double GammaDist::unbounded_cdf(double x) const
{
    return boost::math::cdf(dist,x);
}

inline
double GammaDist::unbounded_icdf(double u) const
{
    return boost::math::quantile(dist,u);
}

inline
double GammaDist::unbounded_pdf(double x) const
{
    return boost::math::pdf(dist,x);
}

inline
double GammaDist::compute_llh_const() const
{
    return -kappa*log(mean/kappa)-lgamma(kappa);
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

template<class IterT>
void GammaDist::append_params(IterT& p) const 
{ 
    *p++ = mean;
    *p++ = kappa;
} 

template<class IterT>
void GammaDist::set_params(IterT& p) 
{ 
    double mean_val = *p++;
    double kappa_val = *p++;
    check_params(mean_val,kappa_val);
    mean = mean_val;
    kappa = kappa_val;
    dist = DistT(kappa,mean/kappa);
    llh_const = compute_llh_const();
}     

inline
void GammaDist::check_params(double mean_val, double kappa_val) 
{ 
    if(mean_val<=0 || !std::isfinite(mean_val)) {
        std::ostringstream msg;
        msg<<"GammaDist::set_params: got bad mean value:"<<mean_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
    if(kappa_val<=0 || !std::isfinite(kappa_val)) {
        std::ostringstream msg;
        msg<<"GammaDist::set_params: got bad kappa value:"<<kappa_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_GAMMADIST_H */
