/** @file GammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief GammaDist class declaration and templated methods
 * 
 */
#ifndef PRIOR_HESSIAN_GAMMADIST_H
#define PRIOR_HESSIAN_GAMMADIST_H

#include <cmath>
#include <random>

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/TruncatedDist.h"

namespace prior_hessian {

/** @brief Single parameter beta distribution where \alpha = \beta, leading to symmetric bounded distribution.  
 * 
 */
class GammaDist : public UnivariateDist
{
    /* These paramter vectors are constant sized, but are handled by accessor functions
     * so as to work generically together with multivariate distributions.
     */
    static const StringVecT _param_names; //Cannonical names for parameters
    static const VecT _param_lbound; //Lower bound on valid parameter values 
    static const VecT _param_ubound; //Upper bound on valid parameter values
public:
    /* Static constant member data */
    static const StringVecT& param_names()  { return _param_names; }
    static const VecT& param_lbound()  { return _param_lbound; }
    static const VecT& param_ubound()  { return _param_ubound; }

    /* Static member functions */
    static constexpr IdxT num_params() { return 2; }
    static bool check_params(double shape, double scale);    /* Check a vector of parameters is valid (in bounds) */    
    static bool check_params(VecT &params);    /* Check a vector of parameters is valid (in bounds) */    
    template<class IterT>
    static bool check_params_iter(IterT &params);    /* Check a vector of parameters is valid (in bounds) */    
    
    GammaDist(double scale=1.0, double shape=1.0);
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {_scale, _shape}; }
    void set_params(const VecT &p) 
    { 
        _scale = checked_scale(p[0]);  
        _shape = checked_shape(p[1]); 
    }
    bool operator==(const GammaDist &o) const { return _scale == o._scale && _shape == o._shape; }
    bool operator!=(const GammaDist &o) const { return !this->operator==(o);}

    double scale() const { return _scale; }
    double shape() const { return _shape; }
    void set_scale(double val) { _scale = checked_scale(val); }
    void set_shape(double val) { _shape = checked_shape(val); }
        
    double mean() const { return _shape*_scale; }
    double median() const { return icdf(0.5); }
        
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
private:
    using RngDistT = std::gamma_distribution<double>; //Used for RNG
    
    static double checked_scale(double val);
    static double checked_shape(double val);

    double _scale; //distribution scale
    double _shape; //distribution shape
    double llh_const;    

    double compute_llh_const() const;
};

/* A bounded gamma dist uses the TruncatedDist adaptor */
using BoundedGammaDist = TruncatedDist<GammaDist>;

inline
BoundedGammaDist make_bounded_gamma_dist(double scale, double shape, double lbound, double ubound)
{
    return {GammaDist(scale, shape),lbound,ubound};
}

namespace detail
{
    template<class Dist>
    class dist_adaptor_traits;
    
    template<>
    class dist_adaptor_traits<GammaDist> {
    public:
        using bounds_adapted_dist = BoundedGammaDist;
        
        static constexpr bool adaptable_bounds = false;
    };
    
    template<>
    class dist_adaptor_traits<BoundedGammaDist> {
    public:
        using bounds_adapted_dist = BoundedGammaDist;
        
        static constexpr bool adaptable_bounds = true;
    };
} /* namespace detail */

inline
bool GammaDist::check_params(double param0, double param1)
{
    return std::isfinite(param0) && std::isfinite(param1) && param0>0 && param1>0;   
}

inline
bool GammaDist::check_params(VecT &params)
{ 
    return params.is_finite() && params[0]>0 && params[1]>0; 
}

template<class IterT>
bool GammaDist::check_params_iter(IterT &p)
{ 
    double scale = *p++;
    double shape = *p++;
    return check_params(scale,shape);
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

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_GAMMADIST_H */
