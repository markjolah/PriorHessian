/** @file mvn_cdf.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief
 * 
 */
#include "PriorHessian/mvn_cdf.h"

#include <cmath>
#include <limits>

#include <armadillo>

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>


namespace {
    const double sqrt2 = sqrt(2.);
    const double inv_sqrt2 = 1./sqrt2;
    const double inv_2pi = 1./(2.*arma::datum::pi);
}

namespace prior_hessian {


/** area of the lower tail of the unit normal curve below t. */
double unit_normal_cdf( double t )
{
    if(t==-INFINITY) return 0;
    if(t==INFINITY) return 1;
    return std::max(std::min(.5*std::erfc(-t*::inv_sqrt2),1.0),0.0);
}

double unit_normal_icdf( double u )
{
    if(u<=0) return -INFINITY;
    if(u>=1) return INFINITY;
    return -::sqrt2*boost::math::erfc_inv(2*u);
}

double bounded(double x)
{
    return std::min(std::max(x,0.),1.);
}

/* Integrates the between y(x)=0 and y(x) = a*x, and right of h, to infinity
 * 
 * when h=0 this is the triangle from the x-axis with angle arctan(a)
 * h - location of vertical line to integrate to the right from
 * a - slope of line above the x-axis
 * gh - normcdf(h);
 */
double owen_t_integral(double h, double a, double gh)
{
    //Check pre-conditions
    if(std::isnan(h)) throw ParameterValueError("a is NaN");
    if(!(gh >=0 && gh<=1)) throw ParameterValueError("gh is not in [0,1]");
    //Edge cases
    if(h == INFINITY) return 0;
    if(std::isnan(a)) throw ParameterValueError("a is NaN");
    if(a == 0) return 0;
    if(h == 0) return std::atan(a)*::inv_2pi;
    if(a == INFINITY) return  (h>0) ? .5*(1-gh) : .5*gh;
    if(a==1) return .5*gh*(1-gh); //This formula works for h and -h by symmetry
    //Use owens 2.4 and 2.5 to handle negative h and a
    if(a<0 && h<0) return -owen_t_integral(-h,-a,1-gh);
    if(a<0) return -owen_t_integral(h,-a,gh);
    if(h<0) return owen_t_integral(-h,a,1-gh);
    
    assert(h>0 && std::isfinite(h));
    assert(a>0 && std::isfinite(a));
    //Use Owens 2.3 to invert a if >1.  This requires an additional normcdf call.
    if(a>1) {
        double ah = a*h;
        double gah = unit_normal_cdf(ah);
        double v1 = owen_t_integral(ah,1/a,gah);
        return .5*(gh+gah) -gh*gah - v1;
    }
    
    assert(a>0 && a<1);
    
    const int max_iter = 1000;
    const double eps = 1E-10;
    double theta = std::atan(a);
    double asq = a*a;
    double asq_powj = asq;
    double h2 = h*h/2;
    double e2h = std::exp(-h2);
    if(e2h==0) return 0;
    double Qj = 1;
    double Sj = Qj;
    double s = 1 - e2h; //First term of series.
    for(int j=1; j<max_iter; j++) {
        Qj *= h2/j;
        Sj += Qj;
        double sign = (j%2==0) ? 1 : -1;
        double v = sign/(2*j+1) * (1-e2h*Sj) * asq_powj;
        s += v;
        if(fabs(v/s) < eps) return ::inv_2pi*(theta - a*s);
        asq_powj *= asq;
    }
    std::ostringstream msg;
    msg<<"Power series failed to converge in max_iter="<<max_iter<<" iterations. h="<<h<<" a="<<a<<" sum="<<s;
    throw RuntimeConvergenceError(msg.str());
}

double owen_b_integral(double h,double k, double r)
{
    if(std::isnan(h)) throw ParameterValueError("h is NaN");
    if(std::isnan(k)) throw ParameterValueError("k is NaN");
    if(fabs(r)>1 || !std::isfinite(r)) throw ParameterValueError("r is not in interval [-1,1]");
    
    if(h==-INFINITY || k==-INFINITY) return 0;
    if(h==INFINITY && k==INFINITY) return 1;
    if(fabs(r)==1) { //Degenerate case.
        if(r>0) return unit_normal_cdf(std::min(h,k));
        else return unit_normal_cdf(h) - unit_normal_cdf(-k);
    }
    assert(fabs(r)<1);
    double sigma = sqrt(1-r*r);
    double gh = unit_normal_cdf(h);
    double ah = (k-r*h)/(h*sigma);
    double bint;
    if(h==k) {
        if(h==0.) return .25+asin(r)*::inv_2pi;
        bint = gh - 2*owen_t_integral(h,ah,gh);
    } else if(h==-k) { 
        bint = .5 - 2*owen_t_integral(h,ah,gh);
    } else {
        double gk = unit_normal_cdf(k);
        double ak = (h-r*k)/(k*sigma);
        double t1 = owen_t_integral(h,ah,gh);
        double t2 = owen_t_integral(k,ak,gk);
        bint = .5*(gh+gk) - t1 - t2;
    }
    if( h*k < 0 || (h*k == 0 && (h<0 || k<0))) bint -= .5;
    return std::min(std::max(bint,0.0),1.0);
}




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
double donnelly_bvn_integral( double ah, double ak, double r )
{
    static double eps = 1.0E-15;
    if (r<-1 || r>1) throw ParameterValueError("must have -1<=rho<=1");
    if ((ah==INFINITY) || (ak==INFINITY)) return 0;
    if ((ah==-INFINITY) && (ak==-INFINITY)) return 1;
    
    double gh = unit_normal_cdf(-ah)/2.0;
    double gk = unit_normal_cdf(-ak)/2.0;

    if(r == 0.0) return bounded(4*gh*gk);

    double b = 0.0; //return value
    double rr = (1+r)*(1-r);
    if(rr==0) { //Degenerate cases r=-1 r=0 r=1
        if(r<0) {
            if(ah+ak<0) b = 2*(gh+gk)-1;
        } else {
            if(ah-ak<0) b = 2*gk;
            else b = 2*gh;
        }
        return bounded(b);
    }

    double sigma_inv = 1./sqrt(rr);
    double con = arma::datum::pi * eps;
    double wh;
    double wk;
    double gw;
    int is;
    if(ah == 0) {
        if(ak == 0) return bounded(0.25+asin(r)*::inv_2pi); // (0,0)
        //(0, !0)
        b = gk;
        wh = -ak;
        wk = (ah/ak - r)*sigma_inv;
        gw = 2*gk;
        is = 1;
    } else {
        //  (!0, 0)
        b = gh;             
        wh = -ah;
        wk = (ak/ah - r)*sigma_inv;
        gw = 2*gh;
        is = -1;
        if(ak != 0) { // ( !0, !0)
            b = gh+gk;
            if (ah*ak < 0) b-= 0.5;
        }
    }
//     std::cout<<"NEW ah:"<<ah<<" ak:"<<ak<<" rho:"<<r<<" gh:"<<gh<<" gk:"<<gk<<"\n";
//     std::cout<<"NEW b:"<<b<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";
    for ( ; ; ) {
        double sgn = -1.0;
        double t = 0.0;
//         std::cout<<"n b:"<<b<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";
        if(wk != 0) {
            if(fabs(wk) == 1) {
                t = .5*wk*gw*(1-gw);
                b += sgn*t;
            } else {
                if (fabs(wk) > 1) {
                    sgn = -sgn;
                    wh *= wk;
                    double g2 = unit_normal_cdf(wh);
                    wk = 1./wk;
                    if(wk < 0) b += 0.5;
                    b += -.5*(gw+g2) + gw*g2;
//                     std::cout<<"nn b: "<<b<<" gw:"<<gw<<" g2:"<<g2<<" t:"<<t<<" sgn:"<<sgn<<" wh:"<<wh<<" wk:"<<wk<<"\n";
                }
                double h2 = wh*wh;
                double a2 = wk*wk;
                double h4 = .5*h2;
                double ex = exp(-h4);
                double w2 = h4*ex;
                double ap = 1.0;
                double s2 = ap - ex;
                double sp = ap;
                double s1 = 0.0;
                double sn = s1;
                double conex = fabs(con/wk);

                for ( ; ; ) {
                    double cn = ap*s2/(sn+sp);
                    s1 += cn;
//                     std::cout<<"NEW cn: "<<cn<<" s1:"<<s1<<" conex:"<<conex<<" sn:"<<sn<<" sp:"<<sp<<" s2:"<<s2<<" w2"<<w2<<" ap:"<<ap<<"\n";

                    if(fabs(cn) <= conex) break;

                    sn = sp;
                    sp += 1.0;
                    s2 -= w2;
                    w2 *= h4/sp;
                    ap *= -a2;
                }
                t = (atan(wk) - wk*s1) * ::inv_2pi;
                b += sgn*t;
            }
        }
        //Check for convergence
//         std::cout<<"N b:"<<b<<" is:"<<is<<" ak:"<<ak<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";
        if(0 <= is || ak == 0) return bounded(b);
        wh = -ak;
        wk = (ah/ak - r)*sigma_inv;
        gw = 2*gk;
        is = 1;
    }
    //never get here
}

double donnelly_bvn_integral_orig( double ah, double ak, double r )
{
    using std::max;
    using std::min;
    using std::fabs;
  double a2;
  double ap;
  double b;
  double cn;
  double con;
  double conex;
  double ex;
  double g2;
  double gh;
  double gk;
  double gw;
  double h2;
  double h4;
  int i;
  static int idig = 15;
  int is;
  double rr;
  double s1;
  double s2;
  double sgn;
  double sn;
  double sp;
  double sqr;
  double t;
  static double twopi = 6.283185307179587;
  double w2;
  double wh;
  double wk;

  b = 0.0;

  gh = unit_normal_cdf ( - ah ) / 2.0;
  gk = unit_normal_cdf ( - ak ) / 2.0;

  if ( r == 0.0 )
  {
    b = 4.00 * gh * gk;
    b = max ( b, 0.0 );
    b = min ( b, 1.0 );
    return b;
  }

  rr = ( 1.0 + r ) * ( 1.0 - r );

  if ( rr < 0.0 )
  {
    return -1;
  }

  if ( rr == 0.0 )
  {
    if ( r < 0.0 )
    {
      if ( ah + ak < 0.0 )
      {
        b = 2.0 * ( gh + gk ) - 1.0;
      }
    }
    else
    {
      if ( ah - ak < 0.0 )
      {
        b = 2.0 * gk;
      }
      else
      {
        b = 2.0 * gh;
      }
    }
    b = max ( b, 0.0 );
    b = min ( b, 1.0 );
    return b;
  }

  sqr = sqrt ( rr );

  if ( idig == 15 )
  {
    con = twopi * 1.0E-15 / 2.0;
  }
  else
  {
    con = twopi / 2.0;
    for ( i = 1; i <= idig; i++ )
    {
      con = con / 10.0;
    }
  }
//
//  (0,0)
//
  if ( ah == 0.0 && ak == 0.0 )
  {
    b = 0.25 + asin ( r ) / twopi;
    b = max ( b, 0.0 );
    b = min ( b, 1.0 );
    return b;
  }
//
//  (0,nonzero)
//
  else if ( ah == 0.0 && ak != 0.0 )
  {
    b = gk;
    wh = -ak;
    wk = ( ah / ak - r ) / sqr;
    gw = 2.0 * gk;
    is = 1;
  }
//
//  (nonzero,0)
//
  else if ( ah != 0.0 && ak == 0.0 )
  {
    b = gh;
    wh = -ah;
    wk = ( ak / ah - r ) / sqr;
    gw = 2.0 * gh;
    is = -1;
  }
//
//  (nonzero,nonzero)
//
  else
  {
    b = gh + gk;
    if ( ah * ak < 0.0 )
    {
      b = b - 0.5;
    }
    wh = - ah;
    wk = ( ak / ah - r ) / sqr;
    gw = 2.0 * gh;
    is = -1;
  }
//     std::cout<<"OLD ah:"<<ah<<" ak:"<<ak<<" rho:"<<r<<" gh:"<<gh<<" gk:"<<gk<<"\n";
//     std::cout<<"OLD b:"<<b<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";
  for ( ; ; )
  {
    sgn = -1.0;
    t = 0.0;
//      std::cout<<"O b:"<<b<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";

    if ( wk != 0.0 )
    {
      if ( fabs ( wk ) == 1.0 )
      {
        t = wk * gw * ( 1.0 - gw ) / 2.0;
        b = b + sgn * t;
      }
      else
      {
        if ( 1.0 < fabs ( wk ) )
        {
          sgn = -sgn;
          wh = wh * wk;
          g2 = unit_normal_cdf ( wh );
          wk = 1.0 / wk;

          if ( wk < 0.0 )
          {
            b = b + 0.5;
          }
          b = b - ( gw + g2 ) / 2.0 + gw * g2;
//             std::cout<<"oo b: "<<b<<" gw:"<<gw<<" g2:"<<g2<<" t:"<<t<<" sgn:"<<sgn<<" wh:"<<wh<<" wk:"<<wk<<"\n";

        }
        h2 = wh * wh;
        a2 = wk * wk;
        h4 = h2 / 2.0;
        ex = exp ( - h4 );
        w2 = h4 * ex;
        ap = 1.0;
        s2 = ap - ex;
        sp = ap;
        s1 = 0.0;
        sn = s1;
        conex = fabs ( con / wk );

        for ( ; ; )
        {
          cn = ap * s2 / ( sn + sp );
          s1 = s1 + cn;
//           std::cout<<"OLD cn: "<<cn<<" s1:"<<s1<<" conex:"<<conex<<" sn:"<<sn<<" sp:"<<sp<<" s2:"<<s2<<" w2"<<w2<<" ap:"<<ap<<"\n";
          if ( fabs ( cn ) <= conex )
          {
            break;
          }
          sn = sp;
          sp = sp + 1.0;
          s2 = s2 - w2;
          w2 = w2 * h4 / sp;
          ap = - ap * a2;
        }
        t = ( atan ( wk ) - wk * s1 ) / twopi;
        b = b + sgn * t;
      }
    }
//     std::cout<<"0 b:"<<b<<" is:"<<is<<" ak:"<<ak<<" wh:"<<wh<<" wk:"<<wk<<" gw:"<<gw<<" is:"<<is<<"\n";
        
    if ( 0 <= is )
    {
      break;
    }
    if ( ak == 0.0 )
    {
      break;
    }
    wh = -ak;
    wk = ( ah / ak - r ) / sqr;
    gw = 2.0 * gk;
    is = 1;
  }

  b = max ( b, 0.0 );
  b = min ( b, 1.0 );

  return b;
}



} /* namespace prior_hessian */
