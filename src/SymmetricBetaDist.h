/** @file SymmetricBetaDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief SymmetricBetaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_SYMMETRICBETADIST_H
#define _PRIOR_HESSIAN_SYMMETRICBETADIST_H
#include <trng/beta_dist.hpp>
#include "UnivariateDist.h"
namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class SymmetricBetaDist : public UnivariateDist<SymmetricBetaDist>
{

public:
    SymmetricBetaDist(double beta, std::string var_name);
    SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name);
    SymmetricBetaDist(double beta, double lbound, double ubound, 
                                std::string var_name, StringVecT&& param_desc);
    
    constexpr static IdxT num_params();
    static StringVecT make_default_param_desc(std::string var_name);
    static double compute_llh_const(double beta);

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
    double convert_to_unitary_coords(double x) const;

    using RNGDistT = trng::beta_dist<double>;
    double beta; //symmetric shape parameter
    RNGDistT dist;
    double llh_const; //Constant term of log-likelihood
    double bound_dist; //distance between bounds
};

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, std::string var_name) :
        UnivariateDist<SymmetricBetaDist>(0,1,var_name,make_default_param_desc(var_name)),
        beta(beta),
        dist(beta,beta),
        llh_const(compute_llh_const(beta)),
        bound_dist(1)
{
}

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, double lbound, double ubound, std::string var_name) :
        UnivariateDist<SymmetricBetaDist>(lbound,ubound,var_name,make_default_param_desc(var_name)),
        beta(beta),
        dist(beta,beta),
        llh_const(compute_llh_const(beta)),
        bound_dist(ubound-lbound)
{
}

inline
SymmetricBetaDist::SymmetricBetaDist(double beta, double lbound, double ubound, 
                                     std::string var_name, StringVecT&& param_desc) :
        UnivariateDist<SymmetricBetaDist>(lbound,ubound,var_name,std::move(param_desc)),
        beta(beta),
        dist(beta,beta),
        llh_const(compute_llh_const(beta)),
        bound_dist(ubound-lbound)
{
}

constexpr
IdxT SymmetricBetaDist::num_params()
{ 
    return 1; 
}

inline
StringVecT SymmetricBetaDist::make_default_param_desc(std::string var_name)
{
    return {std::string("beta_") + var_name};
}

inline
double SymmetricBetaDist::compute_llh_const(double beta)
{
    return -2*lgamma(beta)-lgamma(2*beta);
}

inline
double SymmetricBetaDist::cdf(double x) const
{
    return dist.cdf(convert_to_unitary_coords(x));
}

inline
double SymmetricBetaDist::pdf(double x) const
{
    return dist.pdf(convert_to_unitary_coords(x));
}

inline
double SymmetricBetaDist::llh(double x) const
{
    return rllh(convert_to_unitary_coords(x)) + llh_const;
}

inline
double SymmetricBetaDist::rllh(double x) const
{
    double u = convert_to_unitary_coords(x);
    return (beta-1)*log(u*(1-u));
}

inline
double SymmetricBetaDist::grad(double x) const
{
    double u = convert_to_unitary_coords(x);
    return (beta-1)*(1/u - 1/(1-u));
}

inline
double SymmetricBetaDist::grad2(double x) const
{
    double u = convert_to_unitary_coords(x);
    return (beta-1)*(-1/square(u) - 1/square(1-u));
}

inline
void SymmetricBetaDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double u = convert_to_unitary_coords(x);
    double bm1 = beta-1;
    g  += bm1*(1/u - 1/(1-u));
    g2 += bm1*(-1/square(u) - 1/square(1-u));
}

inline
double SymmetricBetaDist::convert_to_unitary_coords(double x) const
{
    return (x-_lbound)/bound_dist;
}

/* Templated method definitions */

template<class IterT>
void SymmetricBetaDist::insert_params(IterT& p) const 
{ 
    *p++ = beta;
} 

template<class IterT>
void SymmetricBetaDist::set_params(IterT& p) 
{ 
    beta = *p++;
    dist = RNGDistT(beta,beta);
    llh_const = compute_llh_const(beta);
}     

template<class RngT> 
double SymmetricBetaDist::sample(RngT &rng) 
{ 
    return dist(rng);
}
    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_SYMMETRICBETADIST_H */
