/** @file GammaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief GammaDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_GAMMADIST_H
#define _PRIOR_HESSIAN_GAMMADIST_H

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

public:
    static const StringVecT param_names;
    static constexpr IdxT num_params() { return 2; }
    
    GammaDist(double scale=1.0, double shape=1.0);
    
    double get_param(int idx) const;
    void set_param(int idx, double val);
    VecT params() const { return {_scale, _shape}; }
    void set_params(const VecT &p) 
    { 
        _scale = check_scale(p[0]);  
        _shape = check_shape(p[1]); 
    }
    bool operator==(const GammaDist &o) const { return _scale == o._scale && _shape == o._shape; }
    bool operator!=(const GammaDist &o) const { return !this->operator==(o);}

    double scale() const { return _scale; }
    double shape() const { return _shape; }
    void set_scale(double val) { _scale = check_scale(val); }
    void set_shape(double val) { _shape = check_shape(val); }
        
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
protected:
    using RngDistT = std::gamma_distribution<double>; //Used for RNG
    
    static double check_scale(double val);
    static double check_shape(double val);

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

#endif /* _PRIOR_HESSIAN_GAMMADIST_H */
