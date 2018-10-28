/** @file ParetoDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief ParetoDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_PARETODIST_H
#define PRIOR_HESSIAN_PARETODIST_H

#include <cmath>

#include "PriorHessian/UnivariateDist.h"

namespace prior_hessian {

/** @brief Pareto dist with infinite upper bound.
 * 
 */
class ParetoDist : public UnivariateDist
{
    /* These paramter vectors are constant sized, but are handled by accessor functions
     * so as to work generically together with multivariate distributions.
     */
    static const StringVecT _param_names; //Cannonical names for parameters
    static const VecT _param_lbound; //Lower bound on valid parameter values 
    static const VecT _param_ubound; //Upper bound on valid parameter values
public:
    /* Static constant member data */
    static const StringVecT& param_names() { return _param_names; }
    static const VecT& param_lbound() { return _param_lbound; }
    static const VecT& param_ubound() { return _param_ubound; }

    /* Static member functions */
    static constexpr IdxT num_params() { return 2; }
    static bool check_params(double min, double alpha); /* Check parameters are valid (in bounds) */    
    static bool check_params(VecT &params); /* Check a vector of parameters is valid (in bounds) */    
    static bool check_lbound(double min); /* Check the lbound (min) parameter */    
    
    ParetoDist(double min=1.0, double alpha=1.0);
    ParetoDist(const VecT &params);
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {lbound(),_alpha}; }
    void set_params(double min, double alpha) 
    { 
        set_lbound(checked_min(min)); 
        _alpha = checked_alpha(alpha); 
    }
    void set_params(const VecT &p) 
    { 
        set_lbound(checked_min(p[0])); 
        _alpha = checked_alpha(p[1]); 
    }
    bool operator==(const ParetoDist &o) const { return lbound()==o.lbound() && _alpha == o._alpha; }
    bool operator!=(const ParetoDist &o) const { return !this->operator==(o);}

    double alpha() const { return _alpha; } 
    void set_min(double val) { set_lbound(checked_min(val)); }
    void set_alpha(double val) { _alpha = checked_alpha(val); }
    
    void set_lbound(double lbound);
    
    double mean() const { return (_alpha <=1) ? INFINITY : _alpha*lbound()/(_alpha-1); }
    double median() const { return lbound() * std::pow(2,1/_alpha); }
    
    double cdf(double x) const;
    double icdf(double u) const;
    double pdf(double x) const;
    double llh(double x) const { return rllh(x) + llh_const; }
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;
    
    template<class RngT>
    double sample(RngT &rng) const;

     /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);   
    
    template<class IterT>
    void set_params_iter(IterT &params);
   
private:
    static double checked_min(double val);
    static double checked_alpha(double val);

    double _alpha; //distribution shape
    double llh_const;    

    double compute_llh_const() const;
};

inline
bool ParetoDist::check_params(double param0, double param1)
{
    return std::isfinite(param0) && (param0 > 0) &&
           std::isfinite(param1) && (param1 > 0);     
}

inline
bool ParetoDist::check_params(VecT &params)
{ 
    return std::isfinite(params[0]) && (params[0] > 0) &&
           std::isfinite(params[1]) && (params[1] > 0); 
}

template<class IterT>
bool ParetoDist::check_params_iter(IterT &p)
{ 
    double p0 = *p++;
    double p1 = *p++;
    return check_params(p0,p1);
}

inline
double ParetoDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return lbound();
        case 1:
            return _alpha;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

inline
void ParetoDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_min(val);
            return;
        case 1:
            set_alpha(val);
            return;
        default:
            //Don't handle indexing errors.
            return;
    }
}

inline
void ParetoDist::set_lbound(double lbound)
{ 
    set_lbound_internal(lbound);
    llh_const = compute_llh_const();  //Pareto llh_const depends on lbound.
}

inline
bool ParetoDist::check_lbound(double min)
{
    return std::isfinite(min) && min>0;
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


template<class IterT>
void ParetoDist::set_params_iter(IterT &params)
{
    double min = *params++;
    double alpha = *params++;
    return set_params(min,alpha);
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_PARETODIST_H */
