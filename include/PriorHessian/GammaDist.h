/** @file GammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief GammaDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_GAMMADIST_H
#define PRIOR_HESSIAN_GAMMADIST_H

#include <cassert>
#include <cmath>
#include <random>

#include "PriorHessian/Meta.h"
#include "PriorHessian/UnivariateDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class GammaDist : public UnivariateDist
{
    static constexpr IdxT _num_params = 2;
public:
    using NparamsVecT = arma::Col<double>::fixed<_num_params>;

    /* Static member functions */
    static constexpr IdxT num_params() { return _num_params; }
    static constexpr double lbound() { return 0; }
    static constexpr double ubound() { return INFINITY; }
    static bool in_bounds(double u) { return  lbound() < u && u < ubound(); }
    
    static const StringVecT& param_names()  { return _param_names; }
    static const NparamsVecT& param_lbound() { return _param_lbound; }
    static const NparamsVecT& param_ubound() { return _param_ubound; }

    static bool check_params(double shape, double scale);    /* Check  parameters is valid (in bounds) */
    template<class Vec>
    static bool check_params(const Vec &params);    /* Check a vector of parameters is valid (in bounds) */    
    
    GammaDist(double scale, double shape);
    GammaDist() : GammaDist(1.0,1.0) { }
    template<class Vec, meta::ConstructableIfNotSelfT<Vec,GammaDist> = true>
    explicit GammaDist(const Vec &params) : GammaDist(params(0),params(1)) { }
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    NparamsVecT params() const { return {_scale, _shape}; }
    template<class Vec>
    void set_params(const Vec &p) { set_params(p(0),p(1)); }
    void set_params(double scale, double shape);
    bool operator==(const GammaDist &o) const { return _scale == o._scale && _shape == o._shape; }
    bool operator!=(const GammaDist &o) const { return !this->operator==(o);}

    double scale() const { return _scale; }
    double shape() const { return _shape; }
    void set_scale(double val);
    void set_shape(double val);
        
    double mean() const { return _shape*_scale; }
    double median() const { return icdf(0.5); }
        
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
    using RngDistT = std::gamma_distribution<double>; //Used for RNG
    
    /* These paramter vectors are constant sized, but are handled by accessor functions
     * so as to work generically together with multivariate distributions.
     */
    static const StringVecT _param_names; //Cannonical names for parameters
    static const NparamsVecT _param_lbound; //Lower bound on valid parameter values 
    static const NparamsVecT _param_ubound; //Upper bound on valid parameter values

    static double checked_scale(double val);
    static double checked_shape(double val);

    double _scale; //distribution scale
    double _shape; //distribution shape
    
    //Lazy computation of llh_const.  Most use-cases do not need it.
    mutable double llh_const;
    mutable bool llh_const_initialized;
    void initialize_llh_const() const;
    static double compute_llh_const(double shape, double scale);
};

inline
bool GammaDist::check_params(double param0, double param1)
{
    return std::isfinite(param0) && std::isfinite(param1) && param0>0 && param1>0;   
}

template<class Vec>
bool GammaDist::check_params(const Vec &params)
{ 
    return params.is_finite() && params(0)>0 && params(1)>0; 
}

inline
double GammaDist::get_param(int idx) const
{ 
    switch(idx){
        case 0:
            return _scale;
        case 1:
            return _shape;
        default:
            //Don't handle indexing errors.
            return std::numeric_limits<double>::quiet_NaN();
    }
}

inline
void GammaDist::set_param(int idx, double val)
{ 
    switch(idx){
        case 0:
            set_scale(val);
            return;
        case 1:
            set_shape(val);
            return;
        default:
            //Don't handle indexing errors.
            return;
    }
}

inline
double GammaDist::rllh(double x) const
{
    return (_shape-1)*log(x) - x/_scale;
}

inline
double GammaDist::grad(double x) const
{
    return (_shape-1)/x - 1/_scale;
}

inline
double GammaDist::grad2(double x) const
{
    return -(_shape-1)/(x*x);
}

inline
void GammaDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double km1 = _shape-1;
    g  += km1/x - 1/_scale;
    g2 += -km1/(x*x);
}

template<class RngT>
double GammaDist::sample(RngT &rng) const
{
    RngDistT d(_shape,_scale);
    return d(rng);
}

/* Protected methods */
template<class IterT>
bool GammaDist::check_params_iter(IterT &p)
{ 
    double scale = *p++;
    double shape = *p++;
    return check_params(scale,shape);
}

template<class IterT>
void GammaDist::set_params_iter(IterT &p)
{ 
    double scale = *p++;
    double shape = *p++;
    set_params(scale,shape);
}


} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_GAMMADIST_H */
