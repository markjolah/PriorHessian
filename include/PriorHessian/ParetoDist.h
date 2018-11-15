/** @file ParetoDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief ParetoDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_PARETODIST_H
#define PRIOR_HESSIAN_PARETODIST_H

#include <cmath>

#include "PriorHessian/Meta.h"
#include "PriorHessian/UnivariateDist.h"

namespace prior_hessian {

/** @brief Pareto dist with infinite upper bound.
 * 
 */
class ParetoDist : public UnivariateDist
{
    static constexpr IdxT _num_params = 2;
public:
    using NparamsVecT = arma::Col<double>::fixed<_num_params>;
    static constexpr IdxT num_params() { return _num_params; }
    static constexpr double global_lbound() { return 0; }
    static constexpr double ubound() { return INFINITY; }
    bool in_bounds(double u) const { return  lbound() < u && u < ubound(); }

    /* Static constant member data */
    static const StringVecT& param_names() { return _param_names; }
    static const NparamsVecT& param_lbound() { return _param_lbound; }
    static const NparamsVecT& param_ubound() { return _param_ubound; }

    /* Static member functions */
    static bool check_params(double min, double alpha); /* Check parameters are valid (in bounds) */    
    template<class Vec>
    static bool check_params(const Vec &params) { return check_params(params(0),params(1)); } /* Check a vector of parameters is valid (in bounds) */    
    static bool check_lbound(double min); /* Check the lbound (min) parameter */    

    ParetoDist() : ParetoDist(1.0,1.0) { }
    ParetoDist(double min, double alpha);
    template<class Vec, meta::ConstructableIfNotSelfT<Vec,ParetoDist> = true>
    explicit ParetoDist(const Vec &params) : ParetoDist(params(0),params(1)) { }
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    NparamsVecT params() const { return {lbound(),alpha()}; }
    void set_params(double min, double alpha);
    template<class Vec>
    void set_params(const Vec &p) { set_params(p(0),p(1)); }
    bool operator==(const ParetoDist &o) const { return min()==o.min() && alpha() == o.alpha(); }
    bool operator!=(const ParetoDist &o) const { return !this->operator==(o); }

    double alpha() const { return _alpha; } 
    double min() const { return _min; } 
    void set_min(double val);
    void set_alpha(double val);
    
    double lbound() const { return _min; }
    void set_lbound(double lbound);
    
    double mean() const;
    double median() const;
    
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

     /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);   
    
    template<class IterT>
    void set_params_iter(IterT &params);
   
private:
    static const StringVecT _param_names; //Cannonical names for parameters
    static const NparamsVecT _param_lbound; //Lower bound on valid parameter values 
    static const NparamsVecT _param_ubound; //Upper bound on valid parameter values

    static double checked_min(double val);
    static double checked_alpha(double val);

    double _min;
    double _alpha; //distribution shape

    //Lazy computation of llh_const.  Most use-cases do not need it.
    mutable double llh_const;
    mutable bool llh_const_initialized;
    void initialize_llh_const() const;
    static double compute_llh_const(double lbound, double alpha);
};

inline
bool ParetoDist::check_params(double param0, double param1)
{
    return std::isfinite(param0) && (param0 > 0) &&
           std::isfinite(param1) && (param1 > 0);     
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
bool ParetoDist::check_lbound(double min)
{
    return std::isfinite(min) && min>0;
}

inline
double ParetoDist::cdf(double x) const
{
    return 1-pow(lbound()/x, alpha());
}

inline
double ParetoDist::icdf(double u) const
{
    return lbound() / pow(1-u,1/alpha());
}

inline
double ParetoDist::pdf(double x) const
{
    return _alpha/x * pow(lbound()/x,_alpha);
}

inline
double ParetoDist::rllh(double x) const
{
    return -(alpha()+1)*log(x);
}

inline
double ParetoDist::grad(double x) const
{
    return -(alpha()+1)/x;
}

inline
double ParetoDist::grad2(double x) const
{
    return (alpha()+1)/(x*x);
}

inline
void ParetoDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double x_inv = 1/x;
    double ap1ox = (alpha()+1)*x_inv;
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
