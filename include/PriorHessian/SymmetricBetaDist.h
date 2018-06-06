/** @file SymmetricBetaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief SymmetricBetaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_SYMMETRICBETADIST_H
#define _PRIOR_HESSIAN_SYMMETRICBETADIST_H

#include <boost/math/distributions/beta.hpp>

#include "PriorHessian/ScaledFiniteDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class SymmetricBetaDist : public ScaledFiniteDist<SymmetricBetaDist>
{
public:
    SymmetricBetaDist();
    SymmetricBetaDist(double beta, std::string var_name);
    SymmetricBetaDist(double beta, std::string var_name, StringVecT&& param_desc);
    SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name);
    SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();

    double get_param(int idx) const;
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;
    
protected:
    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params_iter(IterT& p);   
    
    double compute_llh_const() const;
    double unscaled_cdf(double x) const;
    double unscaled_icdf(double u) const;
    double unscaled_pdf(double x) const;

    using DistT = boost::math::beta_distribution<double>;
    /* Member variables */
    double beta; //symmetric shape parameter
    DistT dist; //Boost distribution for use in computations of cdf,pdf,icdf

    /* static methods */
    static StringVecT make_default_param_desc(std::string var_name);
    static void check_params(double beta_val);

    /* Friends! */
    friend UnivariateDist<SymmetricBetaDist>;
    friend ScaledFiniteDist<SymmetricBetaDist>;
    friend TruncatingDist<SymmetricBetaDist>;
    template<class RngT> friend class CompositeDist;
};

inline
SymmetricBetaDist::SymmetricBetaDist() :
    SymmetricBetaDist(1,0,1,"x",make_default_param_desc("x"))
{ }

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, std::string var_name) :
    SymmetricBetaDist(beta,0,1,var_name,make_default_param_desc(var_name))
{ }

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, std::string var_name, StringVecT&& param_desc) :
    SymmetricBetaDist(beta,0,1,var_name,std::move(param_desc))
{ }

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name) :
        SymmetricBetaDist(beta,lbound,ubound,var_name,make_default_param_desc(var_name))
{ }

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name, StringVecT&& param_desc) :
        ScaledFiniteDist<SymmetricBetaDist>(lbound, ubound, var_name,std::move(param_desc)),
        beta(beta),
        dist(beta,beta)
{
    this->set_bounds(lbound,ubound);
    this->llh_const = compute_llh_const();
}

constexpr
IdxT SymmetricBetaDist::num_params()
{ 
    return 1; 
}

inline
double SymmetricBetaDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return beta;
        default:
            std::ostringstream msg;
            msg<<"Bad parameter index: "<<idx<<" max:"<<num_params();
            throw IndexError(msg.str());
    }
}


inline
StringVecT SymmetricBetaDist::make_default_param_desc(std::string var_name)
{
    return {std::string("beta_") + var_name};
}

inline
double SymmetricBetaDist::unscaled_cdf(double x) const
{
    return boost::math::cdf(dist,x);
}

inline
double SymmetricBetaDist::unscaled_icdf(double u) const
{
    return boost::math::quantile(dist,u);
}

/**
 * 
 * @param x double in range [0,1].
 */
inline
double SymmetricBetaDist::unscaled_pdf(double x) const
{
    return boost::math::pdf(dist,x);
}

inline
double SymmetricBetaDist::compute_llh_const() const
{
    return -2*lgamma(beta) - lgamma(2*beta);//log(1/Beta(beta,beta))
}

inline
double SymmetricBetaDist::rllh(double x) const
{
    double z = this->convert_to_unitary_coords(x); //Normalized to [0,1]
    return (beta-1) * log(z*(1-z));
}

inline
double SymmetricBetaDist::grad(double x) const
{
    double z = this->convert_to_unitary_coords(x); //Normalized to [0,1]
    return (beta-1) * (1/z-1/(1-z));
}

inline
double SymmetricBetaDist::grad2(double x) const
{
    double z = this->convert_to_unitary_coords(x); //Normalized to [0,1]
    double v= 1/(1-z);
    return (beta-1) * (v*v - 1/(z*z));
}

inline
void SymmetricBetaDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double z = this->convert_to_unitary_coords(x); //Normalized to [0,1]
    double v= 1/(1-z); 
    double bm1 = beta-1;
    g  += bm1*(1/z-v); //(beta-1)*(1/z-1/(1-z))
    g2 += bm1*(v*v-1/(z*z));
}

template<class IterT>
void SymmetricBetaDist::append_params(IterT& p) const 
{ 
    *p++ = beta;
} 

template<class IterT>
void SymmetricBetaDist::set_params_iter(IterT& p) 
{ 
    double beta_val = *p++;
    check_params(beta_val);
    beta = beta_val;
    dist = DistT(beta,beta);  //Reset internal boost dist
    llh_const = compute_llh_const(); //Recompute constants
}

inline
void SymmetricBetaDist::check_params(double beta_val) 
{ 
    if(beta_val<=0 || !std::isfinite(beta_val)) {
        std::ostringstream msg;
        msg<<"SymmetricBetaDist::set_params: got bad beta value:"<<beta_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
}
        
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_SYMMETRICBETADIST_H */
