/** @file util.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 10-2017
 * @brief Utilities and namespace globals
 * 
 * 
 */
#ifndef _PHES_UTIL_H
#define _PHES_UTIL_H

/**
 * 
 * 
 */
namespace phess
{
    using VecT = arma::Col<double>;
    using MatT = arma::Mat<double>;


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
#endif /* _PHES_UTIL_H */
