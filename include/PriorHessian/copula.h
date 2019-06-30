/** @file copula.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief CopulaDist base class.
 */
#ifndef PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H
#define PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H
#include<limits>
namespace prior_hessian {

class ArchimedeanCopula {

protected:
    struct D_GenTerms {
        double log_dn_gen_t;
        double eta_n_np1_t;
    };
    
    struct D_IGenTerms {
        double d1_igen_ui;
        double ieta_21_ui;
    };
    
    struct D2_GenTerms : public D_GenTerms {
        double xi_n_t;
    };
    
    struct D2_IGenTerms : public D_IGenTerms {
        double d2_igen_ui;
        double ixi_1_ui;
    };

    struct DTheta_GenTerms {
        double log_dn_gen_t;
        double eta_0n_1n_t;
    };

    struct DTheta_IGenTerms {
        double sum_log_d1_igen_u;
        double d_10_t;
        double sum_ieta_01_11_u;
    };

    struct D2Theta_GenTerms : public DTheta_GenTerms {
        double xi_0n_t;
    };

    struct D2Theta_IGenTerms : public DTheta_IGenTerms {
        double d_20_t;
        double sum_ixi_01_u;
    };
};
    
    
    
class CopulaParameterError : public std::logic_error
{};
    
template<int dim> //Add SFINAE to ensure dim>=2
class AMHCopula
{
    double theta;
    static bool check_theta(double theta_)
    { 
        return theta_ >= theta_lbound && theta_ <= theta_ubound; 
    }
            
public:
    //Valid domain theta \in [-1,1)
    static const double theta_lbound=-1;
    static const double theta_ubound=std::nextafter(1,-1);//1-epsilon
    
    AMHCopula(double theta_)
    {
        if(!check_theta(theta_)) {
            std::ostringstram msg;
            msg<<"Bad parameter theta: "<<theta_<<" should be in range (inclusive) ["<<theta_lbound<<","<<theta_ubdound<<"]";
            throw CopulaParameterError(msg.str());
        }
        theta = theta_;
    }
    
    void set_theta(double theta_) 
    {
        if(!check_theta(theta_)) {
            std::ostringstram msg;
            msg<<"Bad parameter theta: "<<theta_<<" should be in range (inclusive) ["<<theta_lbound<<","<<theta_ubdound<<"]";
            throw CopulaParameterError(msg.str());
        }
        theta = theta_;
    }
    
    double get_theta() const
    { return theta; }
    
    double gen(double t) const 
    { return (1-theta)/(exp(t)-theta); }
    
    double igen(double u) const 
    { return log((1-theta)/u+theta); }
    
    double igen_sum(const VecT &u) const
    {
        double s=0.;
        for(int k=0;k<dim;k++) s+=igen(u(k));
        return s;
    }
    
    
    double d_gen(int n, double t) const
    {
        
    }
    double d_igen(int n, double t) const;
    
    double d_gen_ratio(int n, int k, double t) const;
    double d_igen_ratio(int n, int k, double u) const;
    
    double cdf(const VecT &u) const;
    { return gen(igen_sum(u)); }
    
    double pdf(const VecT &u) const
    { 
        double p = ddim_gen(igen_sum(u));
        for(int k=0;k<dim;k++) p*=d1_igen(u(k));
        return p;
    }
    
    double llh(const VecT &u) const;
    { 
        double L=log(fabs(ddim_gen(igen(u))));  //Use fabs instead of pow(-1,dim).  Should be faster.
        for(int k=0;k<dim;k++) L+=log(-d1_igen(u(k)));
        return L;
    }
    
    VecT grad(const VecT &u) const
    {
        double z = igrad_sum(u);
        double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
        VecT grad(dim);
        for(int k=0;k<dim;k++)
            grad(k)=d1_igen(u(k))*Q_G1 + ieta21(u(k));
        return grad;
    }
    
    VecT grad2(const VecT &u) const;
    {
        double z = igrad_sum(u);
        double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
        double Q_G2 = eta_np2(z) - Q_G1^2; //eta^{n+2}_n(z) - (eta^{n+1}_n(z))^2
        VecT grad2(dim);
        for(int k=0;k<dim;k++)
            grad2(k) = square(d1_igen(u(k)))*Q_G2 + d2_igen(u(k))*Q_G1 + ieta31(u(k)) - square(ieta21(u(k)));
        
        return grad2;
    }
    
    MatT hess(const VecT &u) const;
    {
        double z = igrad_sum(u);
        double Q_G1 = eta_np1(z); //eta^{n+1}_n(z)
        double Q_G2 = eta_np2(z) - Q_G1^2; //eta^{n+2}_n(z) - (eta^{n+1}_n(z))^2
        VecT d1_igen_u(dim);
        VecT d2_igen_u(dim);
        VecT ieta_21_u(dim); // ieta^{2}_{1}(u_k)
        VecT ieta_31_u(dim); // ieta^{3}_{1}(u_k)
        for(int k=0;k<dim;k++) {
            d1_igen_u(k) = d1_igen(u(d));
            d2_igen_u(k) = d2_igen(u(d));
            ieta_21_u(k) = ieta_21(u(d));
            ieta_31_u(k) = ieta_31(u(d));
        }
        
        MatT hess(dim,dim,arma::fill::zeros);
        //Diagonal: Grad2
        for(int k=0;k<dim;k++) 
            hess(k,k) = square(d1_igen_u(k))*Q_G2 + d2_igen_u(k)*Q_G1 +ieta_31_u(k) - square(ieta_21_u(k));
        //Off-diagonal
        for(int i=0;i<dim-1;i++) for(int j=i+1;j<dim;j++){
            hess(i,j) = d1_igen_u(i)*d1_igen_u(j)*Q_G2;
        }
        return hess;
    }
    
    void llh_grad_accumulate(const VecT &u, double &llh, VecT &grad) const;
    void llh_grad_grad2_accumulate(const VecT &u, double &llh, VecT &grad, VecT &grad2) const;
    void llh_grad_hess_accumulate(const VecT &u, double &llh, VecT &grad, MatT &hess) const;   
}
template<int dim>
double AMHCopula::gen(double u) const
{


template<int dim>
double AMHCopula::cdf(const VecT &u) const

    
    
    
template<int dim> //Add SFINAE to ensure dim>=2
class ClaytonCopula
{
    double theta;
    void check_theta();
public:
    ClaytonCopula(double theta_);
    
    void set_theta(double theta_);
    double get_theta() const;
    
    double gen(double t) const;
    double igen(double u) const;
    
    double d_gen(int n, double t) const;
    double d_igen(int n, double t) const;
    
    double d_gen_ratio(int n, int k, double t) const;
    double d_igen_ratio(int n, int k, double u) const;
    
    double cdf(const VecT &u) const;
    double pdf(const VecT &u) const;
    double llh(const VecT &u) const;
    VecT grad(const VecT &u) const;
    VecT grad2(const VecT &u) const;
    MatT hess(const VecT &u) const;
    
    void llh_grad_accumulate(const VecT &u, double &llh, VecT &grad) const;
    void llh_grad_grad2_accumulate(const VecT &u, double &llh, VecT &grad, VecT &grad2) const;
    void llh_grad_hess_accumulate(const VecT &u, double &llh, VecT &grad, MatT &hess) const;   
}


/** Compute log(1+exp(x)) 
 */
double log1pexp(double x) 
{
    if (x <= -37) return exp(x);
    else if (x <= 18) return log1p(exp(x));
    else if (x <= 33.3) return x + exp(-x);
    else return x;
}

/** Compute log(1-exp(x)) 
 */
double log1mexp(double x) 
{
    static constexpr const double ln2 = log(2);
    if (x <= 0) return -inf;
    else if (x <= ln2) return log(-expm1(-a));
    else return log1p(-exp(-a));
}

} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H */
