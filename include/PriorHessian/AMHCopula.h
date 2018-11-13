/** @file AMCopula.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief Ali-Mikhail0-Haq Archemedian Copula
 */

#ifndef PRIOR_HESSIAN_AMHCOPULA_H
#define PRIOR_HESSIAN_AMHCOPULA_H
#include<limits>

#include "PriorHessian/PriorHessianError.h"

namespace prior_hessian {
    
template<int Ndim> //Add SFINAE to ensure Ndim>=2
class AMHCopula
{
    //Valid domain theta \in [-1,1)
    static const double _default_theta;
    static const double _theta_lbound;
    static const double _theta_ubound;
    static const StringVecT _param_names;
public:
    using NdimVecT = arma::Col<double>::fixed<Ndim>;
    using NdimMatT = arma::Mat<double>::fixed<Ndim,Ndim>;
    static const StringVecT& param_names() { return _param_names; }
    static constexpr IdxT num_params() { return 1; } 
    static constexpr IdxT num_dim() { return Ndim; } 
    static double param_lbound() { return _theta_lbound; }
    static double param_ubound() { return _theta_ubound; }
    static bool check_theta(double val);
    
    AMHCopula() : AMHCopula(_default_theta) {};
    AMHCopula(double theta);

    double theta() const { return _theta; }
    void set_theta(double val);

    bool operator==(const AMHCopula<Ndim> &o) const { return theta() == o.theta(); }
    bool operator!=(const AMHCopula<Ndim> &o) const { return theta() != o.theta(); }
    
    template<class Vec>
    static bool check_params(const Vec &params) { return check_theta(params(0)); }
    template<class Vec>
    void set_params(const Vec &params) { set_theta(params(0)); }
    template<class IterT>
    static bool check_params_iter(IterT &params) { return check_theta(*params++); }
    template<class IterT>
    void append_params(IterT &params) { *params++ = theta(); }
    template<class IterT>
    void set_params_iter(IterT &params) { set_theta(*params++); }

    /* Numerical methods */
    double gen(double t) const;
    double igen(double u) const;
    template<class Vec>
    double igen_sum(const Vec &u) const;
    double d_gen(int n, double t) const;
    double d1_igen(int n, double t) const;    
    double d_gen_ratio(int n, int k, double t) const;
    double d_igen_ratio(int n, int k, double u) const;
    
    template<class Vec>
    double cdf(const Vec &u) const;
    template<class Vec>
    double pdf(const Vec &u) const;
    template<class Vec>
    double llh(const Vec &u) const;
    template<class Vec>
    NdimVecT grad(const Vec &u) const;
    template<class Vec>
    NdimVecT grad2(const Vec &u) const;
    template<class Vec>
    NdimMatT hess(const Vec &u) const;
    template<class Vec, class Vec2>
    void llh_grad_accumulate(const Vec &u, double &llh, Vec2 &grad) const;
    template<class Vec, class Vec2>
    void llh_grad_grad2_accumulate(const Vec &u, double &llh, Vec2 &grad, Vec2 &grad2) const;
    template<class Vec, class Vec2, class Mat>
    void llh_grad_hess_accumulate(const Vec &u, double &llh, Vec2 &grad, Mat &hess) const;   
private:
    double _theta;
};

template<int Ndim>
const double AMHCopula<Ndim>::_default_theta = 0;
template<int Ndim>
const double AMHCopula<Ndim>::_theta_lbound = -1;
template<int Ndim>
const double AMHCopula<Ndim>::_theta_ubound = std::nextafter(1,-1);//1-epsilon
template<int Ndim>
const StringVecT AMHCopula<Ndim>::_param_names = {std::string("theta")};


template<int Ndim>
bool AMHCopula<Ndim>::check_theta(double val)
{ 
    return  _theta_lbound <= val && val <= _theta_ubound; //<=1-epsilon implies <1 for upper bound
}
    
template<int Ndim>
AMHCopula<Ndim>::AMHCopula(double theta)
{ 
    set_theta(theta); 
}

template<int Ndim>
void AMHCopula<Ndim>::set_theta(double val)
{
    if(!check_theta(val)) {
        std::ostringstream msg;
        msg<<"Bad parameter theta: "<<val<<" should be in range (inclusive) ["<<_theta_lbound<<","<<_theta_ubound<<"]";
        throw ParameterValueError(msg.str());
    }
    _theta = val;
}

template<int Ndim>
double AMHCopula<Ndim>::gen(double t) const 
{ return (1-_theta) / (exp(t)-_theta); }
    
template<int Ndim>
double AMHCopula<Ndim>::igen(double u) const 
{ return log((1-_theta)/u + _theta); }
    
template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::igen_sum(const Vec &u) const
{
    double s=0.;
    for(int k=0;k<Ndim;k++) s+=igen(u(k));
    return s;
}
    
    
template<int Ndim>
double AMHCopula<Ndim>::d_gen(int n, double t) const
{
    return 0;
}

template<int Ndim>
double AMHCopula<Ndim>::d1_igen(int n, double t) const
{
    return 0;
}

template<int Ndim>
double AMHCopula<Ndim>::d_gen_ratio(int n, int k, double t) const
{
    return 0;
}

template<int Ndim>
double AMHCopula<Ndim>::d_igen_ratio(int n, int k, double u) const
{
    return 0;
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::cdf(const Vec &u) const  
{ 
    return gen(igen_sum(u)); 
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::pdf(const Vec &u) const
{ 
    double p = ddim_gen(igen_sum(u));
    for(int k=0;k<Ndim;k++) p*=d1_igen(u(k));
    return p;
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::llh(const Vec &u) const
{ 
    double L=log(fabs(ddim_gen(igen(u))));  //Use fabs instead of pow(-1,Ndim).  Should be faster.
    for(int k=0;k<Ndim;k++) L+=log(-d1_igen(u(k)));
    return L;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimVecT 
AMHCopula<Ndim>::grad(const Vec &u) const
{
//     double z = igrad_sum(u);
//     double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
    NdimVecT grad;
//     for(int k=0;k<Ndim;k++) grad(k) = d1_igen(u(k))*Q_G1 + ieta21(u(k));
    return grad;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimVecT 
AMHCopula<Ndim>::grad2(const Vec &u) const
{
//     double z = igrad_sum(u);
//     double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
//     double Q_G2 = eta_np2(z) - Q_G1^2; //eta^{n+2}_n(z) - (eta^{n+1}_n(z))^2
    NdimVecT grad2;
//     for(int k=0;k<Ndim;k++)  grad2(k) = square(d1_igen(u(k)))*Q_G2 + d2_igen(u(k))*Q_G1 + ieta31(u(k)) - square(ieta21(u(k)));
    return grad2;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimMatT 
AMHCopula<Ndim>::hess(const Vec &u) const
{
//     double z = igrad_sum(u);
//     double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
//     double Q_G2 = eta_np2(z) - Q_G1^2; //eta^{n+2}_n(z) - (eta^{n+1}_n(z))^2
//     NdimVecT d1_igen_u;
//     NdimVecT d2_igen_u;
//     NdimVecT ieta_21_u; // ieta^{2}_{1}(u_k)
//     NdimVecT ieta_31_u; // ieta^{3}_{1}(u_k)
//     for(int k=0;k<Ndim;k++) {
//         d1_igen_u(k) = d1_igen(u(d));
//         d2_igen_u(k) = d2_igen(u(d));
//         ieta_21_u(k) = ieta_21(u(d));
//         ieta_31_u(k) = ieta_31(u(d));
//     }
//     
    NdimMatT hess(arma::fill::zeros);
//     //Diagonal: Grad2
//     for(int k=0;k<Ndim;k++) 
//         hess(k,k) = square(d1_igen_u(k))*Q_G2 + d2_igen_u(k)*Q_G1 +ieta_31_u(k) - square(ieta_21_u(k));
//     //Off-diagonal
//     for(int i=0;i<Ndim-1;i++) for(int j=i+1;j<Ndim;j++){
//         hess(i,j) = d1_igen_u(i)*d1_igen_u(j)*Q_G2;
//     }
    return hess;
}    

} /* namespace phess */

#endif /* PRIOR_HESSIAN_AMHCOPULA_H */
