/** @file rng/symmetric_beta_dist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief A C++ random number generator 
 * 
 */
#ifndef _PRIOR_HESSIAN_RNG_SYMMETRICBETADIST_H
#define _PRIOR_HESSIAN_RNG_SYMMETRICBETADIST_H

#include <cmath>
#include <random>
#include <boost/math/special_functions/beta.hpp>

namespace prior_hessian {

namespace rng {
    
template <typename RealT=double>
class symmetric_beta_dist
{
public:
    typedef RealT result_type;

    class param_type {
        RealT _alpha;
        RealT _norm;
    public:
        typedef symmetric_beta_dist distribution_type;
        explicit param_type(RealT alpha = 1.0) : _alpha(alpha), _norm(boost::math::beta(alpha, alpha)) {}
        RealT alpha() const { return _alpha; }
        RealT norm() const { return _norm; }
        bool operator==(const param_type& other) const { return _alpha == other._alpha;}
        bool operator!=(const param_type& other) const { return !(*this == other); }
    };
    
    explicit symmetric_beta_dist(RealT alpha = 2.0): alpha_dist(alpha) { }
    explicit symmetric_beta_dist(const param_type& param): alpha_dist(param.alpha()) { }
    
    void reset() { }
    
    param_type param() const { return param_type(alpha()); }
    
    void param(const param_type& param)
    {
        p = param;
        alpha_dist = GammaDistT(param.alpha());
    }
    
    void set_params(double alpha)
    {
        p = param_type(alpha);
        alpha_dist = GammaDistT(alpha);
    }
    
    template <typename URNG>
    result_type operator()(URNG& engine) { return generate(engine, alpha_dist); }
    
    template <typename URNG>
    result_type operator()(URNG& engine, const param_type& param)
    {
        GammaDistT tmp_alpha_dist(param.alpha());
        return generate(engine, tmp_alpha_dist);
    }
    
    result_type min() const { return 0.0; }
    result_type max() const { return 1.0; }
    
    result_type alpha() const { return p.alpha(); }
    
    void alpha(RealT alpha) 
    { 
        p = param_type(alpha);
        alpha_dist = GammaDistT(alpha);//default theta=1 
    }
    
    bool operator==(const symmetric_beta_dist<RealT>& other) const
    {
        return param() == other.param() && alpha_dist == other.alpha_dist;
    }
    
    bool operator!=(const symmetric_beta_dist<RealT>& other) const { return !(*this == other);}
    
    RealT pdf(RealT x) const 
    {
        if (x<0 or x>1) return 0;
        if ((x==0 or x==1) and p.alpha()-1<0) return std::numeric_limits<RealT>::quiet_NaN();
        return 1/p.norm()*std::pow(x, p.alpha()-1)*std::pow(1-x, p.alpha()-1);
    }
    
    RealT cdf(RealT x) const 
    {
        if (x<=0) return 0;
        if (x>=1) return 1;
        return boost::math::ibeta(p.alpha(), p.alpha(), x); //symmetric alpha args
    }
    
    RealT icdf(RealT x) const 
    {
        if (x<0 or x>1) return std::numeric_limits<RealT>::quiet_NaN();
        if (x==0) return 0;
        if (x==1) return 1;
        return boost::math::ibeta_inv(p.alpha(), p.alpha(), x); //symmetric alpha args
    }
    
private:
    using GammaDistT = std::gamma_distribution<RealT>;
    param_type p;
    GammaDistT alpha_dist;
    
    template <typename URNG>
    result_type generate(URNG& engine, GammaDistT& x_gamma)
    {
        //Symmetric distribution.  Draw both from x_gamma
        result_type x = x_gamma(engine);
        result_type y = x_gamma(engine);
        if (x+y==0.) return 0.;
        else return x/(x+y);
    }
}; /* class symmetric_beta_dist */
    
} /* namespace rng */

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_RNG_SYMMETRICBETADIST_H */
