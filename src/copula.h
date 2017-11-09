/** @file copula.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 10-2017
 * @brief The copula computations
 * 
 * Source for log1pexp and log1mexp computations
 *  Accurately Computing log(1−exp(−|a|)) Assessed by the Rmpfr package. Martin Machler. ETH Zurich April, Oct. 2012
 * 
 */
#ifndef _COPULA_H
#define _COPULA_H

/**
 * 
 * 
 */
namespace phess
{
    using VecT = arma::Col<double>;
    using MatT = arma::Mat<double>;

class FranksCopula
{
    int dim;
    double theta;  //model parameter
    
public:
    FranksCopula(int dim, double theta);
    set_dim(int dim);
    set_theta(double theta);
    
    double gen(double t) const;
    double gen_der(int n, double t)  const;
    double gen_der_ratio(int n,int m, double t)  const;

    double igen(double u) const;
    double igen_der(int n, double u) const;
    double igen_der_ratio(int n,int m, double u) const;
    
    double igen_vec(const VecT &u) const;
    
    double pdf(const VecT &u) const;
    double cdf(const VecT &u) const;
    
    double llh(const VecT &u) const;
    double llh_grad(const VecT &u, double &llh, VecT &grad) const;
    double llh_grad_hess(const VecT &u, double &llh, VecT &grad, MatT &grad_hess) const;   
}

class SymmetricBetaDist
{
    double beta;
    SymmetricBetaDist(double beta);
    static double log_prior_const(double beta);
    double llh();
    
    double rllh(double x);
    void rllh_grad(double x, double &rllh, VecT &grad);
    void 
}



/** Compute log(1+exp(x)) 
 */
double log1pexp(double x) 
{
    if (x <= -37) return exp(x);
    else if (x <= 18) return log1p(exp(x));
    else if (x <= 33.3) return x + exp(-x);
    else return x;
}

/** Compute log(1-exp(x)) 
 */
double log1mexp(double x) 
{
    static constexpr const double ln2 = log(2);
    if (x <= 0) return -inf;
    else if (x <= ln2) return log(-expm1(-a));
    else return log1p(-exp(-a));
}

} /* namespace phess */
#endif /* _COPULA_H */
