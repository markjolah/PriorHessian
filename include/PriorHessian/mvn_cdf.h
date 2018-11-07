/** @file mvn_cdf.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief Numerical computation of multivariate normal cdfs in 2,3 and higher dims.
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_MVN_CDF_H
#define PRIOR_HESSIAN_MVN_CDF_H

#include<iomanip>

#include<random>
#include<cmath>

#include "PriorHessian/util.h"

namespace prior_hessian {

double unit_normal_cdf( double t );
double unit_normal_icdf( double u );

double owen_t_integral(double h, double a, double gh);

inline
double owen_t_integral(double h, double a)
{
    return owen_t_integral(h,a,unit_normal_cdf(h));
}

double owen_b_integral(double h,double k, double r);


/** compute the upper-right tail of the bivariate normal distribution
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
double donnelly_bvn_integral( double ah, double ak, double r );
double donnelly_bvn_integral_orig( double ah, double ak, double r );




template<class Vec, class Mat>
double donnelly_bvn_cdf(const Vec &b, const Mat &sigma)
{
    double s0 = sqrt(sigma(0,0));
    double s1 = sqrt(sigma(1,1));
    double rho = sigma(0,1)/(s0*s1);
    double z0 = b(0)/s0;
    double z1 = b(1)/s1;
    double integral = donnelly_bvn_integral(-z0,-z1,rho);//Itergation of x,y coordinates in bvn_integral is inverted from normal CDF
    return integral;
}

template<class Vec, class Mat>
double owen_bvn_cdf(const Vec &b, const Mat &sigma)
{
    //Normalize by sigma
    double s0 = sqrt(sigma(0,0));
    double s1 = sqrt(sigma(1,1));
    double rho = sigma(0,1)/(s0*s1);
    double z0 = b(0)/s0;
    double z1 = b(1)/s1;
    return owen_b_integral(z0,z1,rho);
}

/** compute the multivariate normal cdf integral
 */    
template<class Vec, class Mat>
double mc_mvn_integral(const Vec &a, const Vec &b, const Mat &U, double &error, int &niter)
{
    int Ndim = a.n_elem;
    const int Nmax = 10000*Ndim;
    const double eps = 1E-4;
    const double alpha = 2.5;
    double c = U(0,0);
    double d1 = unit_normal_cdf(a(0)/c);
    double e1 = unit_normal_cdf(b(0)/c);
    double f1 = e1-d1;
    double int_sum = 0.;
    double var_sum = 0.;
    VecT y(Ndim);
    std::random_device R;
    std::default_random_engine rng{R()};
    std::uniform_real_distribution<double> uniform(0, 1);
    VecT ys(Ndim-1);
    
    error = eps;
    for(niter=0; niter<Nmax; niter++){
        double d = d1;
        double e = e1;
        double f = f1;
        for(int i=1; i<Ndim; i++){
            c = U(i,i);
            ys(i-1) = unit_normal_icdf(d+uniform(rng)*(e-d));
            double q = 0;
            for(int k=0; k<i; k++) q+=ys(k)*U(k,i);
            d = unit_normal_cdf((a(i)-q)/c);
            e = unit_normal_cdf((b(i)-q)/c);
            f *= (e-d);
        }
        double delta = (f-int_sum)/(niter+1);
        int_sum += delta;
        var_sum = (niter-1)*var_sum/(niter+1) + delta*delta;
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
double mc_mvn_cdf_core(const Vec &b, const Mat &U, double &error, int &niter)
{
    int Ndim = b.n_elem;
    const int Nmax = 1000*Ndim;
    const double eps = 1E-5;
    const double alpha = 2.5;
    double c = U(0,0);
    //double d1 = 0;
    double e1 = unit_normal_cdf(b(0)/c);
    double f1 = e1;
    double int_sum = 0.;
    double var_sum = 0.;
    VecT y(Ndim);
    std::random_device R;
    std::default_random_engine rng{R()};
    std::uniform_real_distribution<double> uniform(0, 1);
    VecT ys(Ndim-1);
//     std::cout<<" e1:"<<e1<<" f1:"<<f1<<"\n";
    error = eps;
    for(niter=0; niter<Nmax; niter++){
        double e = e1;
        double f = f1;
        for(int i=1; i<Ndim; i++){
            c = U(i,i);
            ys(i-1) = unit_normal_icdf(uniform(rng)*e);
            double q = 0;
            for(int k=0; k<i; k++) q+=ys(k)*U(k,i);
            e = unit_normal_cdf((b(i)-q)/c);
            f *= e;
        }
        double delta = (f-int_sum)/(niter+1);
        int_sum += delta;
        var_sum = (niter-1)*var_sum/(niter+1) + delta*delta;
        error = alpha * sqrt(var_sum);
//         std::cout<<"n:"<<niter<<" f:"<<f<<" int_sum:"<<int_sum<<" var_sum:"<<var_sum<<" error:"<<error<<"\n";
        if(error < eps) break;
    }
    return int_sum;
}

template<class Vec, class Mat>
double mc_mvn_cdf(const Vec &b, const Mat &S, double &error)
{
    MatT U = arma::chol(S);
    int niter;
    return mc_mvn_cdf_core(b,U,error,niter);
}

namespace genz
{
    namespace fortran {
        //Genz fortran code interface declaration
        /*
        *  Parameters
        *     N      INTEGER, the number of variables.
        *     LOWER  REAL, array of lower integration limits.
        *     UPPER  REAL, array of upper integration limits.
        *     INFIN  INTEGER, array of integration limits flags:
        *            if INFIN(I) < 0, Ith limits are (-infinity, infinity);
        *            if INFIN(I) = 0, Ith limits are (-infinity, UPPER(I)];
        *            if INFIN(I) = 1, Ith limits are [LOWER(I), infinity);
        *            if INFIN(I) = 2, Ith limits are [LOWER(I), UPPER(I)].
        *     CORREL REAL, array of correlation coefficients; the correlation
        *            coefficient in row I column J of the correlation matrix
        *            should be stored in CORREL( J + ((I-2)*(I-1))/2 ), for J < I.
        *            THe correlation matrix must be positive semidefinite.
        *     MAXPTS INTEGER, maximum number of function values allowed. This 
        *            parameter can be used to limit the time. A sensible 
        *            strategy is to start with MAXPTS = 1000*N, and then
        *            increase MAXPTS if ERROR is too large.
        *     ABSEPS REAL absolute error tolerance.
        *     RELEPS REAL relative error tolerance.
        *     ERROR  REAL estimated absolute error, with 99% confidence level.
        *     VALUE  REAL estimated value for the integral
        *     INFORM INTEGER, termination status parameter:
        *            if INFORM = 0, normal completion with ERROR < EPS;
        *            if INFORM = 1, completion with ERROR > EPS and MAXPTS 
        *                           function vaules used; increase MAXPTS to 
        *                           decrease ERROR;
        *            if INFORM = 2, N > 500 or N < 1.
        */
        extern "C" 
        int mvndst_(int *n, double lower[], double upper[],
            int infin[], double correl[], int *maxpts, double *abseps, double *releps, 
            double *error, double *value, int *inform);
    }

    // S = sigma covariamce matrix
    template<class Vec, class Mat>
    double mvn_cdf_genz(const Vec &b, const Mat &S, double &error)
    {
        int maxpts = 10000;
        double abseps = 1E-5;
        double releps = 1E-5;
        int N = b.n_elem;
        VecT lower(N);
        lower.fill(-INFINITY);
        VecT s = arma::sqrt(S.diag());
    
        MatT U = S / (s*s.t());
        VecT upper = b / s;
        //Fill correlation matrix while normalizing eachj column by the sigma on diagonal of U.
        VecT correl(N*(N-1)/2);
        int k=0;
        for(int j=1; j<N; j++) for(int i=0; i<j; i++) correl(k++) = U(i,j);
        arma::Col<int> infin(N,arma::fill::zeros); // infin(i) = 0 implies (-inf, upper(i)] bounds.
        double value;
        int inform=-1;
        fortran::mvndst_(&N, lower.memptr(), upper.memptr(), infin.memptr(), correl.memptr(), &maxpts, &abseps, &releps, &error, &value, &inform);
        return std::min(std::max(value,0.),1.);
    }

} /* namespace prior_hessian::gentz */

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MVN_CDF_H */
