/** @file ParetoDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief ParetoDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_PARETODIST_H
#define _PRIOR_HESSIAN_PARETODIST_H

#include <cmath>

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/UpperTruncatedDist.h"

namespace prior_hessian {

/** @brief Pareto dist with infinite upper bound.
 * 
 */
class ParetoDist : public UnivariateDist
{
public:
    static const StringVecT param_names;
    static constexpr IdxT num_params() { return 1; }
    
    ParetoDist(double alpha=1.0, double lbound=1.0);
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    double alpha() const;
    void set_alpha(double val);
    
    void set_lbound(double lbound);
        
    double cdf(double x) const;
    double icdf(double u) const;
    double pdf(double x) const;
    double llh(double x) const;
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;
    
    template<class RngT>
    double sample(RngT &rng) const;
protected:
    static double check_alpha(double val);
    static double check_lbound(double val);
   
    double _alpha; //distribution shape
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded pareto dist uses the UpperTruncatedDist adaptor */
using BoundedParetoDist = UpperTruncatedDist<ParetoDist>;

BoundedParetoDist make_bounded_pareto_dist(double alpha, double lbound, double ubound)
{
    return {ParetoDist(alpha,lbound),ubound};
}

namespace detail
{
    template<class Dist>
    class dist_adaptor_traits;
    
    template<>
    class dist_adaptor_traits<ParetoDist> {
    public:
        using bounds_adapted_dist = BoundedParetoDist;
        
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    class dist_adaptor_traits<BoundedParetoDist> {
    public:
        using bounds_adapted_dist = BoundedParetoDist;
        
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace detail */

inline
void ParetoDist::set_lbound(double lbound)
{ 
    UnivariateDist::set_lbound(check_lbound(lbound)); 
    llh_const = compute_llh_const();  //Pareto llh_const depends on lbound.
}

inline
double ParetoDist::cdf(double x) const
{
    return 1-pow(lbound()/x,_alpha);
}

inline
double ParetoDist::icdf(double u) const
{
    return lbound() / pow(1-u,1/_alpha);
}

inline
double ParetoDist::pdf(double x) const
{
    return _alpha/x * pow(lbound()/x,_alpha);
}

inline
double ParetoDist::rllh(double x) const
{
    return -(_alpha+1)*log(x);
}

inline
double ParetoDist::grad(double x) const
{
    return -(_alpha+1)/x;
}

inline
double ParetoDist::grad2(double x) const
{
    return (_alpha+1)/(x*x);
}

inline
void ParetoDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double x_inv = 1/x;
    double ap1ox = (_alpha+1)*x_inv;
    g  -= ap1ox ;   // -(alpha+1)/x
    g2 += ap1ox*x_inv;  // (alpha+1)/x^2
}

template<class RngT>
double ParetoDist::sample(RngT &rng) const
{
    std::uniform_real_distribution<double> d;
    double u = 1-d(rng); // u is uniform on (0,1]
    return lbound()/pow(u,1/_alpha) ;
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_PARETODIST_H */
