/** @file AMCopula.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief Ali-Mikhail0-Haq Archimedean Copula
 */

#ifndef PRIOR_HESSIAN_AMHCOPULA_H
#define PRIOR_HESSIAN_AMHCOPULA_H
#include<limits>

#include "PriorHessian/PriorHessianError.h"
#include "PriorHessian/ArchimedeanCopula.h"
#include "PriorHessian/PolyLog.h"

namespace prior_hessian {
    
template<int Ndim> //Add SFINAE to ensure Ndim>=2
class AMHCopula : public ArchimedeanCopula
{
    //Valid domain theta \in [-1,1)
    static const double _default_theta;//0
    static const double _theta_lbound; //-1
    static const double _theta_ubound;//1-epsilon
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


    
    template<class Vec>
    double cdf(const Vec &u) const;
    template<class Vec>
    double pdf(const Vec &u) const;
    template<class Vec>

    double llh(const Vec &u) const;
    template<class Vec>
    double rllh(const Vec &u) const;
    double rllh_const() const;

    template<class Vec>
    NdimVecT grad(const Vec &u) const;
    template<class Vec>
    NdimVecT grad2(const Vec &u) const;
    template<class Vec>
    NdimMatT hess(const Vec &u) const;

    template<class Vec, class Vec2>
    void rllh_grad_accumulate(const Vec &u, double &rllh, Vec2 &grad) const;
    template<class Vec, class Vec2>
    void rllh_grad_grad2_accumulate(const Vec &u, double &rllh, Vec2 &grad, Vec2 &grad2) const;
    template<class Vec, class Vec2, class Mat>
    void rllh_grad_hess_accumulate(const Vec &u, double &rllh, Vec2 &grad, Mat &hess) const;

    /* Derivatives with respect to parameter theta */
    template<class Vec>
    static void rllh_dtheta_accumulate(double theta, const Vec &u, double &rllh, double &dtheta);
    template<class Vec>
    static void rllh_d2theta_accumulate(double theta, const Vec &u, double &rllh, double &dtheta, double &d2theta);

    template<class RngT>
    NdimVecT sample(RngT &rng) const;
    
    /* Public numerical methods */
    double gen(double t) const;
    double ddim_gen(double t) const;
    double igen(double u) const;
    double d1_igen(double u) const;
    template<class Vec>
    double igen_sum(const Vec &u) { return igen_sum(theta(),u); }
private:
    struct PDFTerms {
        double z;
        double prod_d1_igen;
    };

    double _theta;

    /* Core coputational methods:
     * Organized into static methods for the generator and inverse-generator constants
     * for computing the rllh and derivatives vs u and vs theta.
     * 
     * This organization keeps terms with common subexpressions together, and
     * concentrates all of the computational code into a small set of methods.
     *
     * Each method returns a struct from ArchimedeanCopula with the relevent scalars for
     * computing the rllh, and first and second derivatives.
     * 
     */
    template<class Vec>
    static double igen_sum(double theta, const Vec &u);
    template<class Vec>
    static double compute_z(double theta, const Vec &u);
    template<class Vec>
    static PDFTerms compute_pdf_terms(double theta, const Vec &u);
    static double ddim_gen_z(double theta, double z); // z=theta*exp(-t)
    
    template<class Vec>
    static D_GenTerms compute_d_gen_terms(double theta, const Vec &u);
    template<class Vec>
    static D2_GenTerms compute_d2_gen_terms(double theta, const Vec &u);
    template<class Vec>
    static D_GenTerms compute_d_gen_terms_only(double theta, const Vec &u); //Ignore rllh terms
    template<class Vec>
    static D2_GenTerms compute_d2_gen_terms_only(double theta, const Vec &u); //Ignore rllh terms

    static D_IGenTerms compute_d_igen_terms(double theta, double ui);
    static D2_IGenTerms compute_d2_igen_terms(double theta, double ui);

    /* Computation of deriviatives with respect to theta */ 
    template<class Vec>
    static DTheta_GenTerms compute_dtheta_gen_terms(double theta, const Vec &u);
    template<class Vec>
    static D2Theta_GenTerms compute_d2theta_gen_terms(double theta, const Vec &u);

    template<class Vec>
    static DTheta_IGenTerms compute_dtheta_igen_terms(double theta, const Vec &u);
    template<class Vec>
    static D2Theta_IGenTerms compute_d2theta_igen_terms(double theta, const Vec &u);
    

};

/* Templated static member variable definitions */
template<int Ndim>
const double AMHCopula<Ndim>::_default_theta = 0;
template<int Ndim>
const double AMHCopula<Ndim>::_theta_lbound = -1;
template<int Ndim>
const double AMHCopula<Ndim>::_theta_ubound = std::nextafter(1,-1);
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

// template<int Ndim>
// template<class Vec>
// double AMHCopula<Ndim>::cdf_direct(const Vec &u) const  
// { 
//     return gen(igen_sum(u)); 
// }

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::cdf(const Vec &u) const  
{ 
    auto terms = compute_cdf_terms(theta(),u);
    return terms.prod_d1_igen * ddim_gen_z(terms.z);
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::pdf(const Vec &u) const
{ 
    auto terms = compute_pdf_terms(theta(),u);
    return terms.prod_d1_igen * ddim_gen_z(terms.z);
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::llh(const Vec &u) const
{ 
    double L = log(fabs(ddim_gen(igen_sum(u))));  //Use fabs instead of pow(-1,Ndim).  Should be faster.
    for(IdxT k=0;k<Ndim;k++) L += log(-d1_igen(u(k)));
    return L;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimVecT 
AMHCopula<Ndim>::grad(const Vec &u) const
{
    NdimVecT grad2;
    auto gterms = compute_d_gen_terms_only(theta(),u);
    for(int i=0; i<Ndim; i++) {
        auto igterms = compute_d_igen_terms(theta(),u(i));
        grad(i) += igterms.d1_igen_ui * gterms.eta_n_np1_t + igterms.ieta_21_ui;
    }
    return grad;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimVecT 
AMHCopula<Ndim>::grad2(const Vec &u) const
{
    NdimVecT grad2;
    auto gterms = compute_d2_gen_terms_only(theta(),u);
    for(int i=0; i<Ndim; i++) {
        auto igterms = compute_d2_igen_terms_only(theta(),u(i));
        grad2(i) = square(igterms.d1_igen_ui)*gterms.xi_n_t + igterms.d2_igen_ui*gterms.eta_n_np1_t + igterms.ixi_1_ui;
    }
    return grad2;
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::NdimMatT 
AMHCopula<Ndim>::hess(const Vec &u) const
{
    NdimMatT hess;
    auto gterms = compute_d2_gen_terms_only(theta(),u);
    NdimVecT  d1_igen_u;
    for(int i=0; i<Ndim; i++) {
        auto igterms = compute_d2_igen_terms_only(theta(),u(i));
        hess(i,i) = square(igterms.d1_igen_ui)*gterms.xi_n_t + igterms.d2_igen_ui*gterms.eta_n_np1_t + igterms.ixi_1_ui;
        d1_igen_u(i) = igterms.d1_igen_ui;
        for(int j=0; j<i; j++) hess(j,i) = d1_igen_u(i)*d1_igen_u(j)*gterms.xi_n_t;
    }
    return arma::symmatu(hess);
}


/*Public Non-Static computational methods */
template<int Ndim>
double AMHCopula<Ndim>::gen(double t) const 
{ return (1-_theta) / (exp(t)-_theta); }

/* gen^{('Ndim)}(t) */
template<int Ndim>
double AMHCopula<Ndim>::ddim_gen(double t) const
{ 
    double sign_1mtheta = num_dim()%2==0 ? (1-_theta) : (_theta-1);
    double z = _theta*exp(-t);
    return sign_1mtheta/_theta * polylog::polylog<-Ndim>(z); 
}
      
/* gen^{-1}(u) */
template<int Ndim>
double AMHCopula<Ndim>::igen(double u) const 
{ return log((1-_theta)/u + _theta); }

/* (d^1/du^1) gen^{-1}(u) */
template<int Ndim>
double AMHCopula<Ndim>::d1_igen(double u) const
{ 
    double theta_m_1 = _theta-1;
    double W = _theta *u - theta_m_1;
    return theta_m_1 / (u*W);
}


/*Private Static computational methods */
template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::igen_sum(double theta, const Vec &u)
{
    double s=0;
    double one_m_theta = 1-theta;
    for(IdxT k=0;k<Ndim;k++) s += log(one_m_theta/u(k)+theta);
    return s;
}

template<int Ndim>
template<class Vec>
double AMHCopula<Ndim>::compute_z(double theta, const Vec &u)
{
    double z = theta;
    double theta_m_1 = theta-1;
    for(IdxT i=0; i<Ndim; i++) z*= u(i)/(u(i)*theta-theta_m_1);
    return z;
}

template<int Ndim>
double AMHCopula<Ndim>::ddim_gen_z(double theta, double z)
{ 
    double sign_1mtheta = num_dim()%2==0 ? (1-theta) : (theta-1);
    return sign_1mtheta/theta * polylog::polylog<-Ndim>(z); 
}

template<int Ndim>
template<class Vec>
typename AMHCopula<Ndim>::PDFTerms 
AMHCopula<Ndim>::compute_pdf_terms(double theta, const Vec &u)
{
    PDFTerms terms;
    terms.z = theta;
    terms.prod_d1_igen = 1;
    double one_m_theta = 1-theta;
    for(IdxT i=0; i<Ndim; i++) {
        double W_inv = 1/(u(i)*theta+one_m_theta);
        terms.z *= W_inv*u(i);
        terms.prod_d1_igen *= one_m_theta*W_inv/u(i);
    }
    return terms;
}

// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D_GenTerms 
// AMHCopula<Ndim>::compute_d_gen_terms(double theta, const Vec &u)
// {
//     D_GenTerms terms;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     terms.log_dn_gen_t = log(fabs((theta-1)/theta * (z/pow(1-z,Ndim+1)) * ep_nm1_z));
//     terms.eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     return terms;
// }
// 
// /*
//  * Do not computer rllh terms.  Derivitive calculations only
//  * 
//  */
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D_GenTerms 
// void AMHCopula<Ndim>::compute_d_gen_terms_only(double theta, const Vec &u)
// {
//     D_GenTerms terms;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     terms.log_dn_gen_t = NAN; //Do not compute
//     terms.eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     return terms;
// }
// 
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D2_GenTerms 
// void AMHCopula<Ndim>::compute_d2_gen_terms(double theta, const Vec &u)
// {
//     D2_GenTerms terms;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     terms.log_dn_gen_t = log(fabs((theta-1)/theta * (z/pow(1-z,Ndim+1)) * ep_nm1_z));
//     terms.eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     terms.xi_n_t = eulerian_polynomial<Ndim+1>(z) /(square(z-1)*ep_nm1_z) - square(eta_n_np1_t);
//     return terms;
// }
// 
// /*
//  * Do not computer rllh terms.  Derivitive calculations only
//  * 
//  */
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D2_GenTerms 
// void AMHCopula<Ndim>::compute_d2_gen_terms_only(double theta, const Vec &u)
// {
//     D2_GenTerms terms;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     terms.log_dn_gen_t = NAN;  //Do not compute
//     terms.eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     terms.xi_n_t = eulerian_polynomial<Ndim+1>(z) /(square(z-1)*ep_nm1_z) - square(eta_n_np1_t);
//     return terms;
// }
// 
// template<int Ndim>
// typename AMHCopula<Ndim>::D_IGenTerms 
// void AMHCopula<Ndim>::compute_d_igen_terms(double theta, double ui)
// {
//     D_IGenTerms terms;
//     double W = theta*ui - theta + 1;
//     terms.d1_igen_ui =  (theta-1) / (ui*W);
//     terms.ieta_21_ui = -2*theta/W + terms.d1_igen_ui;
//     return terms
// }
// 
// template<int Ndim>
// typename AMHCopula<Ndim>::D2_IGenTerms 
// void AMHCopula<Ndim>::compute_d2_igen_terms(double theta, double ui)
// {
//     D2_IGenTerms terms;
//     double W = theta*ui - theta + 1;
//     double ui_inv = 1/ui;
//     double W_inv = 1/W;
//     double Wui_inv = ui_inv*W_inv;
//     terms.d1_igen_ui =  (theta-1)*Wui_inv;
//     terms.d2_igen_ui =  (theta*u+W)*Wui_inv * terms.d1_igen_ui
//     terms.ixi_1_ui = square(theta*W_inv) + square(ui_inv);
//     terms.ieta_21_ui = -2*theta*W_inv + terms.d1_igen_ui;
//     return terms
// }
// 
// /*
//  * Compute only terms needed for second derivative.
//  * 
//  */
// template<int Ndim>
// typename AMHCopula<Ndim>::D2_IGenTerms 
// void AMHCopula<Ndim>::compute_d2_igen_terms_only(double theta, double ui)
// {
//     D2_IGenTerms terms;
//     double W = theta*ui - theta + 1;
//     double ui_inv = 1/ui;
//     double W_inv = 1/W;
//     terms.d1_igen_ui =  (theta-1)*ui_inv*W_inv;
//     terms.d2_igen_ui =  (theta*u+W)*ui_inv*W_inv * terms.d1_igen_ui
//     terms.ixi_1_ui = square(theta*W_inv) + square(ui_inv);
//     terms.ieta_21_ui = NAN; //Do not compute.  Needed for grad1 only.
//     return terms
// }
// 
// 
// template<int Ndim>
// template<class Vec, class Vec2>
// void AMHCopula<Ndim>::rllh_grad_accumulate(const Vec &u, double &rllh, Vec2 &grad) const
// {
//     auto gterms = compute_d_gen_terms(theta(),u);
//     rllh += gterms.log_dn_gen_t;
//     for(int i=0; i<Ndim; i++) {
//         auto igterms = compute_d_igen_terms(theta(),u(i));
//         rllh += log(-igterms.d1_igen_ui);
//         grad(i) += igterms.d1_igen_ui * gterms.eta_n_np1_t + igterms.ieta_21_ui;
//     }
// }
// 
// template<int Ndim>
// template<class Vec, class Vec2>
// void AMHCopula<Ndim>::rllh_grad_grad2_accumulate(const Vec &u, double &rllh, Vec2 &grad, Vec2 &grad2) const
// {
//     auto gterms = compute_d2_gen_terms(theta(),u);
//     rllh += gterms.log_dn_gen_t;
//     for(int i=0; i<Ndim; i++) {
//         auto igterms = compute_d2_igen_terms(theta(),u(i));
//         rllh += log(-igterms.d1_igen_ui);
//         grad(i) += igterms.d1_igen_ui * gterms.eta_n_np1_t + igterms.ieta_21_ui;
//         grad2(i) += square(igterms.d1_igen_ui)*gterms.xi_n_t + igterms.d2_igen_ui*gterms.eta_n_np1_t + igterms.ixi_1_ui;
//     }
// }
// 
// template<int Ndim>
// template<class Vec, class Vec2, class Mat>
// void AMHCopula<Ndim>::rllh_grad_hess_accumulate(const Vec &u, double &rllh, Vec2 &grad, Mat &hess) const
// {
//     auto gterms = compute_d2_gen_terms(theta(),u);
//     rllh += gterms.log_dn_gen_t;
//     NdimVecT  d1_igen_u;
//     for(int i=0; i<Ndim; i++) {
//         auto igterms = compute_d2_igen_terms(theta(),u(i));
//         rllh += log(-igterms.d1_igen_ui);
//         grad(i) += igterms.d1_igen_ui * gterms.eta_n_np1_t + igterms.ieta_21_ui;
//         hess(i,i) += square(igterms.d1_igen_ui)*gterms.xi_n_t + igterms.d2_igen_ui*gterms.eta_n_np1_t + igterms.ixi_1_ui;
//         d1_igen_u(i) = igterms.d1_igen_ui;
//         for(int j=0; j<i; j++) hess(j,i) += d1_igen_u(i)*d1_igen_u(j)*gterms.xi_n_t;
//     }
// }
// 
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::DTheta_GenTerms 
// void AMHCopula<Ndim>::compute_dtheta_gen_terms(double theta, const Vec &u)
// {
//     DTheta_GenTerms terms;
//     double theta_m_1 = theta-1;
//     double theta_inv = 1/theta;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     double eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     terms.log_dn_gen_t = log(fabs(theta_m_1*theta_inv*(z/pow(1-z,Ndim+1))*ep_nm1_z));
//     terms.eta_0n_1n_t = theta_inv * (eta_n_np1_t + 1/theta_m_1);
//     return terms;
// }
// 
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::DTheta_IGenTerms 
// AMHCopula<Ndim>::compute_dtheta_igen_terms(double theta, const Vec &u)
// {
//     DTheta_IGenTerms terms; //Initialized to 0 in constructor
//     double theta_m_1 = theta-1;
//     for(int i=0; i<Ndim; i++) {
//         double ui = u(i);
//         double W = theta*ui - theta_m_1;  // W(theta, u(i)) = W*u(i) - theta + 1
//         double d1_igen_ui =  theta_m_1/(ui*W); // (d/dt) \widetilde{\phi}(u_i)
//         double d10_igen_ui = (u_i-1)/W; // (d/dtheta) \widetilde{\phi}(u_i)
//         double ieta_01_11_ui = square(W)/d1_igen_ui; // \widetilde{\eta}_{(0,1)}^{(1,1)}(u_i)
//         assert(d1_igen_ui >= 0);
//         terms.sum_log_d1_igen_u += log(-d1_igen_ui);        
//         terms.sum_d10_igen_u += d10_igen_ui;
//         terms.sum_ieta_01_11_u += ieta_01_11_ui;
//     }
//     return terms;
// }
// 
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D2Theta_GenTerms 
// void AMHCopula<Ndim>::compute_d2theta_gen_terms(double theta, const Vec &u)
// {
//     D2Theta_GenTerms terms;
//     double theta_m_1 = theta-1;
//     double theta_inv = 1/theta;
//     double z = compute_z(theta,u); // z(\theta,t) = \theta*\exp{-\sum_{i=1}^n \igen(u_i)}
//     double ep_nm1_z = eulerian_polynomial<Ndim-1>(z);
//     double eta_n_np1_t = eulerian_polynomial<Ndim>(z) / ((z-1)*ep_nm1_z);
//     double xi_n_t = eulerian_polynomial<Ndim+1>(z) /(square(z-1)*ep_nm1_z) - square(eta_n_np1_t);
//     terms.log_dn_gen_t = log(fabs(theta_m_1*theta_inv*(z/pow(1-z,Ndim+1))*ep_nm1_z));
//     terms.eta_0n_1n_t = theta_inv * (eta_n_np1_t + 1/theta_m_1);
//     terms.xi_0n_t = square(theta_inv) * ((1-2*theta)/square(theta_m_1) - eta_n_np1_t + xi_n_t)
//     return terms;
// }
// 
// template<int Ndim>
// template<class Vec>
// typename AMHCopula<Ndim>::D2Theta_IGenTerms 
// AMHCopula<Ndim>::compute_d2theta_igen_terms(double theta, const Vec &u)
// {
//     D2Theta_IGenTerms terms; //Initialized to 0 in constructor
//     double theta_m_1 = theta-1;
//     for(int i=0; i<Ndim; i++) {
//         double ui = u(i);
//         double W = theta*ui - theta_m_1;  // W(theta, u(i)) = W*u(i) - theta + 1
//         double ui_inv = 1/ui;
//         double W_inv = 1/W;
//         double d1_igen_ui =  theta_m_1*ui_inv*W_inv; // (d/dt) \widetilde{\phi}(u_i)
//         double d10_igen_ui = (u_i-1)*W_inv; // (d/dtheta) \widetilde{\phi}(u_i)
//         double ieta_01_11_ui = square(W)/d1_igen_ui; // \widetilde{\eta}_{(0,1)}^{(1,1)}(u_i)
//         double d20_igen_ui = -square(d_10_igen_ui); // (d^2/dtheta^2) \widetilde{\phi}(u_i)
//         double ixi_01_ui = (ui-2*W)/d1_igen_ui; // \widetilde{\xi}_{(0,1)}(u_i)
//         assert(d1_igen_ui >= 0);
//         terms.sum_log_d1_igen_u += log(-d1_igen_ui);        
//         terms.sum_d10_igen_u += d10_igen_ui;
//         terms.sum_ieta_01_11_u += ieta_01_11_ui;
//         terms.sum_d20_igen_u += d20_igen_ui;
//         terms.sum_ixi_01_u += ixi_01_ui;
//     }
//     return terms;
// }
// 
// template<int Ndim>
// template<class Vec>
// void AMHCopula<Ndim>::rllh_dtheta_accumulate(double theta, const Vec &u, double &rllh, double &dtheta)
// {
//     auto gterms = compute_dtheta_gen_terms(theta,u);
//     auto igterms = compute_dtheta_igen_terms(theta,u);
//     rllh += gterms.log_dn_gen_t + igterms.sum_log_d1_igen_u;
//     dtheta += gterms.eta_0n_1n_t*igterms.sum_d10_iget_t + igterms.sum_ieta_01_11_u;
// }
// 
// template<int Ndim>
// template<class Vec>
// void AMHCopula<Ndim>::rllh_d2theta_accumulate(double theta, const Vec &u, double &rllh, double &dtheta, double &d2theta)
// {
//     auto gterms = compute_d2theta_gen_terms(theta,u);
//     auto igterms = compute_d2theta_igen_terms(theta,u);
//     rllh += gterms.log_dn_gen_t + igterms.sum_log_d1_igen_u;
//     dtheta += gterms.eta_0n_1n_t*igterms.sum_d10_iget_t + igterms.sum_ieta_01_11_u;
//     d2theta += gterms.eta_0n_1n_t*igterms.sum_d20_igen_t + gterms.xi_0_n_t*igterms.sum_d10_igen_t + igterms.sum_ixi_0_1_u;
// }

template<int Ndim>
template<class RngT>
typename AMHCopula<Ndim>::NdimVecT 
AMHCopula<Ndim>::sample(RngT &rng) const
{
    return {};
}


} /* namespace phess */

#endif /* PRIOR_HESSIAN_AMHCOPULA_H */
