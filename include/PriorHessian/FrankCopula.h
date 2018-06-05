/** @file FrankCopula.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief The Frank copula computations
 * 
 * 
 */
#ifndef _PHESS_FRANKCOPULA_H
#define _PHESS_FRANKCOPULA_H

/**
 * 
 * 
 */
namespace phess
{

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
};


} /* namespace phess */
#endif /* _PHESS_FRANKCOPULA_H */
