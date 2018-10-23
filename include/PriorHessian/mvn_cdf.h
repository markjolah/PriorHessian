/** @file mvn_cdf.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief Numerical computation of multivariate normal cdfs in 2,3 and higher dims.
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_MVN_CDF_H
#define PRIOR_HESSIAN_MVN_CDF_H

#include<random>
#include<cmath>

namespace prior_hessian {

double unit_normal_cdf( double t );
double unit_normal_icdf( double u );
    
/** compute the bivariate normal cdf integral
 * computes the probability for two normal variates X and Y
 *    whose correlation is R, that AH <= X and AK <= Y.
 * 
 * Adapted to modern C++ with efficiency improvements by:
 * Mark Olah (mjo@cs.unm DOT edu)
 * 10/2018
 * 
 * Reference:
 *    Thomas Donnelly,
 *    Algorithm 462: Bivariate Normal Distribution,
 *    Communications of the ACM,
 *    October 1973, Volume 16, Number 10, page 638.
 */    
double bvn_integral( double ah, double ak, double r );

template<class Vec, class Mat>
double bvn_cdf(const Vec &b, const Mat &sigma)
{
    double rho = sigma(0,1) / sqrt(sigma(0,0)*sigma(1,1));
    return 1. - bvn_integral(b(0),b(1),rho);
}

/** compute the multivariate normal cdf integral
 * computes the probability for two normal variates X and Y
 * 
 * Adapted to modern C++ with efficiency improvements by:
 * Mark Olah (mjo@cs.unm DOT edu)
 * 10/2018
 * 
 * Reference:
 *    Thomas Donnelly,
 *    Algorithm 462: Bivariate Normal Distribution,
 *    Communications of the ACM,
 *    October 1973, Volume 16, Number 10, page 638.
 */    
template<class Vec, class Mat>
double mvn_integral(const Vec &a, const Vec &b, const Mat &U)
{
    const int Nmax = 20;
    const double eps = 1E-9;
    const double alpha = 2.5;
    int Ndim = a.n_elem;
    double c = U(0,0);
    double d = unit_normal_cdf(a(0)/c);
    double e = unit_normal_cdf(b(0)/c);
    double f = e-d;
    double int_sum=0.;
    double var_sum=0.;
    VecT y(Ndim);
    std::random_device R;
    std::default_random_engine rng{R()};
    std::uniform_real_distribution<double> uniform(0, 1);
    double error = eps;
    for(int n=0; n<Nmax; n++){
        double q=0;
        for(int i=1; i<Ndim; i++){
            c = U(i,i);
            double y = unit_normal_icdf(d+uniform(rng)*(e-d));
            q += y*U(i-1,i);
            d = unit_normal_cdf((a(i)-q)/c);
            e = unit_normal_cdf((b(i)-q)/c);
            f *= (e-d);
        }
        double delta = (f-int_sum)/(n+1);
        int_sum += delta;
        var_sum = (n-1)*var_sum/(n+1) + delta*delta;
        error = alpha * sqrt(var_sum);
        if(error < eps) break;
    }
    return int_sum;
}

/**
 * 
 * For the cdf a=-Infinity, so d=0.
 * 
 */
template<class Vec, class Mat>
double mvn_cdf(const Vec &b, const Mat &U)
{
    const int Nmax = 20;
    const double eps = 1E-9;
    const double alpha = 2.5;
    int Ndim = b.n_elem;
    double c = U(0,0);
    //double d = 0.;
    double e = unit_normal_cdf(b(0)/c);
    double f = e;
    double int_sum=0.;
    double var_sum=0.;
    VecT y(Ndim);
    std::random_device R;
    std::default_random_engine rng{R()};
    std::uniform_real_distribution<double> uniform(0, 1);
    double error = eps;
    for(int n=0; n<Nmax; n++){
        double q=0;
        for(int i=1; i<Ndim; i++){
            c = U(i,i);
            double y = unit_normal_icdf(uniform(rng)*e);
            q += y*U(i-1,i);
            e = unit_normal_cdf((b(i)-q)/c);
            f *= e;
        }
        double delta = (f-int_sum)/(n+1);
        int_sum += delta;
        var_sum = (n-1)*var_sum/(n+1) + delta*delta;
        error = alpha * sqrt(var_sum);
        if(error < eps) break;
    }
    return int_sum;
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MVN_CDF_H */
