/** @file MultivariateNormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief MultivariateNormalDist base class.
 */
#ifndef PRIOR_HESSIAN_MULTIVARIATENORMALDIST_H
#define PRIOR_HESSIAN_MULTIVARIATENORMALDIST_H

#include "PriorHessian/MultivariateDist.h"
#include "PriorHessian/Meta.h"
#include "PriorHessian/mvn_cdf.h"

namespace prior_hessian {

/** @brief Multivariate Normal distribution
 * @param Ndim Number of dimensions >=2
 *
 */
template<int Ndim> //, meta::ConstructableIf< (Ndim>=2) > = true>
class MultivariateNormalDist : public MultivariateDist<Ndim> {
private:    
    static constexpr IdxT _num_params= Ndim+(Ndim*Ndim+Ndim)/2;
    
public:
    static constexpr IdxT num_params() {return _num_params;}
    
    using typename MultivariateDist<Ndim>::NdimVecT;
    using typename MultivariateDist<Ndim>::NdimMatT;
    using MultivariateDist<Ndim>::num_dim;
    
    using NparamsVecT = arma::Col<double>::fixed<num_params()>;

    MultivariateNormalDist();    //Default to unit Gaussian
    template<class Vec, class Mat>
    MultivariateNormalDist(Vec &&mu, Mat &&sigma);
    
    template<class Vec>
    static bool check_mu(const Vec &mu);
    template<class Mat>
    static bool check_sigma(const Mat &sigma);
    template<class Vec, class Mat>
    static bool check_params(const Vec &mu, const Mat &sigma);
    template<class Vec>
    static bool check_params(const Vec &params);
       
    const NdimVecT& mu() const;
    const NdimMatT& sigma() const;
    const NdimMatT& sigma_inv() const;
    
    template<class Vec> void set_mu(Vec&& val);
    template<class Mat> void set_sigma(Mat&& val);
    
    bool operator==(const MultivariateNormalDist<Ndim> &o) const;    
    bool operator!=(const MultivariateNormalDist<Ndim> &o) const { return !this->operator==(o); }
    
    static const StringVecT& param_names();
    static const VecT& param_lbound();
    static const VecT& param_ubound();
        
    double get_param(int idx) const;
 
    NparamsVecT params() const;
    template<class Vec>
    void set_params(const Vec &p);

    template<class Vec,class Mat>
    void set_params(Vec &&mu, Mat &&sigma);
    
    /* Import names from Dependent Base Class */    
    VecT mean() const { return mu(); }
    VecT mode() const { return mu(); }
    
    template<class Vec> double cdf(Vec x) const; // Not implemented. Too expensive.
    template<class Vec> double pdf(const Vec &x) const;
    template<class Vec> double llh(const Vec &x) const;
    template<class Vec> double rllh(const Vec &x) const;
    template<class Vec> NdimVecT grad(const Vec &x) const;
    template<class Vec> NdimVecT grad2(const Vec &x) const;
    template<class Vec> NdimMatT hess(const Vec &x) const;
    
    template<class Vec,class Vec2>
    void grad_grad2_accumulate(const Vec &x, Vec2 &g, Vec2 &g2) const;
    template<class Vec,class Vec2,class Mat>
    void grad_hess_accumulate(const Vec &x, Vec2 &g, Mat &hess) const;
    
    template<class RngT>
    NdimVecT sample(RngT &rng) const;
  
// protected:
    /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);

    template<class IterT>
    void append_params(IterT &params) const;

    template<class IterT>
    void set_params_iter(IterT &params);
private:    
//     static constexpr IdxT _num_params= Ndim+(Ndim*Ndim+Ndim)/2;
    static StringVecT _param_names; //Cannonical names for parameters
    static NparamsVecT _param_lbound; //Lower bound on valid parameter values 
    static NparamsVecT _param_ubound; //Upper bound on valid parameter values
    
    template<class Vec>
    static NdimMatT compressed_upper_triangular_to_full_matrix(const Vec &v);

    template<class Mat>
    static VecT full_matrix_to_compressed_upper_triangular(const Mat &m);

    static void idx_to_row_col(int idx, int &row, int &col);
    static bool init_param_names();
    static bool init_param_lbound();
    static bool init_param_ubound();
    
    /* Non-static private members
     */
    NdimVecT _mu;
    NdimMatT _sigma;
    NdimMatT _sigma_inv;
    NdimVecT _lbound;
    NdimVecT _ubound;
    NdimMatT _sigma_chol; //cholesky decoposition of sigma (lower triangular form s.t. A*A.t()=sigma)

    //Lazy computation of llh_const.  Most use-cases do not need it.
    mutable double llh_const;
    mutable bool llh_const_initialized;
    void initialize_llh_const() const;
    static double compute_llh_const(const NdimMatT &sigma);
};

namespace helpers 
{
    template<class Vec, class Mat>
    double compute_quadratic_from_symmetric(IdxT Ndim, const Vec &v, const Mat &A)
    {
        double z=0;
        for(IdxT c=0; c<Ndim; c++) {
            double vc = v(c);
            for(IdxT r=0; r<c; r++) z+= 2*vc*v(r)*A(r,c); //Account for both off-diagonal elements simultantously.
            z+=square(vc)*A(c,c);
        }
        return z;
    }
}


/* Templated static member variables */
template<int Ndim>
StringVecT MultivariateNormalDist<Ndim>::_param_names;

template<int Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::_param_lbound;

template<int Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::_param_ubound;
/* Constructors */

template<int Ndim>
MultivariateNormalDist<Ndim>::MultivariateNormalDist() : 
        MultivariateDist<Ndim>(),
        _mu(arma::fill::zeros), 
        _sigma(arma::fill::zeros)
{
    _sigma.diag().ones();
    _sigma_inv = _sigma;//sigma == sigma_inv == eye(Ndim)
    _sigma_chol = _sigma;
    llh_const_initialized = false;
}
    
template<int Ndim>
template<class Vec, class Mat>
MultivariateNormalDist<Ndim>::MultivariateNormalDist(Vec &&mu, Mat &&sigma) 
{
    set_mu(std::forward<Vec>(mu));
    set_sigma(std::forward<Mat>(sigma));
}
    
/* public static methods */
template<int Ndim>
template<class Vec>
bool MultivariateNormalDist<Ndim>::check_mu(const Vec &mu)
{
    return mu.is_finite();
}

template<int Ndim>
template<class Mat>
bool MultivariateNormalDist<Ndim>::check_sigma(const Mat &sigma)
{
    if(!sigma.is_finite()) return false;
    if(arma::any(sigma.diag()<=0)) return false;
    NdimMatT R; 
    return arma::chol(R,arma::symmatu(sigma));//sigma is upper triangular.
}

template<int Ndim>
template<class Vec, class Mat>
bool MultivariateNormalDist<Ndim>::check_params(const Vec &mu, const Mat &sigma)
{
    return check_mu(mu) && check_sigma(sigma);
}

template<int Ndim>
template<class Vec>
bool MultivariateNormalDist<Ndim>::check_params(const Vec &params)
{
    return check_mu(params.head(Ndim)) && 
            check_sigma(compressed_upper_triangular_to_full_matrix(params.tail(num_params()-Ndim)));
}

template<int Ndim>
template<class IterT>
bool MultivariateNormalDist<Ndim>::check_params_iter(IterT &params)
{
    for(int k = 0; k<Ndim; k++) if( !std::isfinite(*params++)) return false;
    NdimMatT S;
    for(int j=0;j<Ndim;j++) for(int i=0;i<=j;i++) S(i,j) = *params++;
    return check_sigma(S);
}


/* private static methods */

template<int Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimMatT 
MultivariateNormalDist<Ndim>::compressed_upper_triangular_to_full_matrix(const Vec &v)
{
    int Z = v.n_elem;
    if(Z!=num_params()-Ndim) {
        std::ostringstream msg;
        msg<<"Invalid vector size Z:"<<Z<<" should be:"<<num_params()-Ndim;
        throw ParameterSizeError(msg.str());
    }
    NdimMatT m;
    IdxT n=0;
    for(int j=0;j<Ndim;j++) for(int i=0;i<=j; i++) m(i,j) = v(n++);
    for(int j=0;j<Ndim-1;j++) for(int i=j+1;i<Ndim; i++) m(i,j) = m(j,i); //copy upper triangle to lower.
    return m;
}

template<int Ndim>
template<class Mat>
VecT
MultivariateNormalDist<Ndim>::full_matrix_to_compressed_upper_triangular(const Mat &m)
{
    int Z = num_params()-Ndim;
    VecT v(Z);
    auto it = v.begin();
    for(int j=0;j<Ndim;j++) for(int i=0;i<=j;i++) *it++ = m(i,j);
    return v;
}


template<int Ndim>
const StringVecT& MultivariateNormalDist<Ndim>::param_names()
{ 
    static bool _dummy = init_param_names(); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_names; 
}

template<int Ndim>
const VecT& MultivariateNormalDist<Ndim>::param_lbound()
{ 
    static bool _dummy = init_param_lbound(); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_lbound;
}

template<int Ndim>
const VecT& MultivariateNormalDist<Ndim>::param_ubound()
{ 
    static bool _dummy = init_param_ubound(); //Run initialization only once
    (void) _dummy; //Prevent an unsed variable warning
    return _param_ubound;
}

//translate linear idx into covariance matrix sigma using the ordering of an upper-triangular 
// matrix in col-major form.
template<int Ndim>
void  MultivariateNormalDist<Ndim>::idx_to_row_col(int idx, int &row, int &col)
{
    col = 0;
    int s = 0;
    while(s+col < idx) s+= ++col; //Find the colum, keeping truck of the total s of previous elements in previous columns
    row = idx-s;        
}

template<int Ndim>
bool MultivariateNormalDist<Ndim>::init_param_names()
{
    _param_names.reserve(num_params());
    for(int k=0;k<Ndim;k++) {
        std::ostringstream name;
        name<<"mu_"<<k+1;
        _param_names.emplace_back(name.str());
    }
    for(int c=0;c<Ndim;c++) for(int r=0;r<=c;r++) {
        std::ostringstream name;
        name<<"sigma_"<<r+1<<"_"<<c+1;
        _param_names.emplace_back(name.str());
    }
    return true;
}

template<int Ndim>
bool MultivariateNormalDist<Ndim>::init_param_lbound()
{ 
    _param_lbound.fill(-INFINITY);    
    for(IdxT c=1, k=Ndim; k<num_params(); k += ++c) _param_lbound(k)=0; //Diagonal elements of cov are positive
    return true;
}
    
template<int Ndim>
bool MultivariateNormalDist<Ndim>::init_param_ubound()
{ 
    _param_ubound.fill(INFINITY);
    return true;
}
    
/* Non-static methods */

template<int Ndim>
const typename MultivariateNormalDist<Ndim>::NdimVecT& 
MultivariateNormalDist<Ndim>::mu() const 
{ return _mu; }

template<int Ndim>
const typename MultivariateNormalDist<Ndim>::NdimMatT& 
MultivariateNormalDist<Ndim>::sigma() const 
{ return _sigma; }

template<int Ndim>
const typename MultivariateNormalDist<Ndim>::NdimMatT& 
MultivariateNormalDist<Ndim>::sigma_inv() const 
{ return _sigma_inv; }

// template<int Ndim>
// const typename MultivariateNormalDist<Ndim>::NdimMatT& 
// MultivariateNormalDist<Ndim>::sigma_chol() const 
// { return _sigma_chol; }

template<int Ndim>
template<class Vec>
void MultivariateNormalDist<Ndim>::set_mu(Vec&& val) 
{ if(check_mu(val)) _mu = std::forward<Vec>(val); }

template<int Ndim>
template<class Mat>
void MultivariateNormalDist<Ndim>::set_sigma(Mat&& val) 
{ 
    //Mat is upper-triangular symmetric, positive definite.
    if(!val.is_finite()) throw ParameterValueError("Sigma matrix is not-finite.");
    if(arma::any(val.diag()<=0)) throw ParameterValueError("Sigma matrix is not positive definite.");
    try{
        _sigma_chol = arma::chol(arma::symmatu(val),"lower");
    } catch (std::runtime_error &e) {
        throw ParameterValueError("Cholesky decomposition failure. Sigma is not positive definite.");
    }
    try {
        _sigma_inv = arma::inv_sympd(arma::symmatu(val));
    } catch (std::logic_error &e) {
        std::ostringstream msg;
        msg<<"Bad sigma size: "<<val.n_rows<<","<<val.n_cols<<"\n";
        throw ParameterSizeError(msg.str());
    } catch (std::runtime_error &e) {
        throw ParameterValueError("Sigma is not symmetric positive semi-definite with bounded eigenvalues.  Numerical inversion failure.");
    } 
    _sigma = arma::symmatu(std::forward<Mat>(val)); 
    llh_const_initialized = false;
}

template<int Ndim>
bool MultivariateNormalDist<Ndim>::operator==(const MultivariateNormalDist<Ndim> &o) const 
{ 
    return arma::all(mu() == o.mu()) && arma::all(arma::all(sigma() == o.sigma())); 
}

template<int Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::params() const
{
    NparamsVecT p;
    p.head(Ndim) = mu();
    p.tail(num_params()-Ndim) = full_matrix_to_compressed_upper_triangular(sigma());
    return p;
}

template<int Ndim>
double MultivariateNormalDist<Ndim>::get_param(int idx) const
{
    if(idx<Ndim) return _mu[idx];
    //otherwise pram is a sigma index as an upper-triangular matrix in col-major form.
    idx -= Ndim;
    int row,col;
    idx_to_row_col(idx,row,col);
    return _sigma(row,col);
}

template<int Ndim>
template<class Vec>
void MultivariateNormalDist<Ndim>::set_params(const Vec &p)
{ 
    set_mu(p.head(Ndim));
    set_sigma(compressed_upper_triangular_to_full_matrix(p.tail(num_params()-Ndim)));
}

template<int Ndim>
template<class Vec,class Mat>
void MultivariateNormalDist<Ndim>::set_params(Vec &&mu_val, Mat &&sigma_val)
{ 
    set_mu(std::forward<Vec>(mu_val));
    set_sigma(std::forward<Mat>(sigma_val));
}

template<int Ndim>
template<class IterT>
void MultivariateNormalDist<Ndim>::set_params_iter(IterT &params)
{
    NdimVecT m;
    std::copy_n(params,Ndim,m.begin());
    params+=Ndim;
    NdimMatT S;
    for(int j=0;j<Ndim;j++) for(int i=0;i<=j;i++) S(i,j) = *params++;
    set_params(std::move(m),std::move(S));
}

template<int Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::cdf(Vec x) const
{
    VecT z = x-mu();
    double error;
    return genz::mvn_cdf_genz(z, sigma(), error);
}

template<>
template<class Vec>
double MultivariateNormalDist<2>::cdf(Vec x) const
{
    return owen_bvn_cdf((x-mu()).eval(), sigma());
}

template<int Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::pdf(const Vec &x) const
{
    // ((2*pi)^Ndim * det(Sigma))^{-1/2} * exp(-1/2*(x-mu)'*Sigma^{-1}*(x-mu))
//     NdimVecT delta = x-mu();
//     return 1./sqrt(arma::det(sigma)) * pow(2*arma::datum::pi,-Ndim/2.) * exp(-.5*delta.t()*arma::inv(sigma)*delta);
    return exp(llh(x));
}

template<int Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::llh(const Vec &x) const
{
    if(!llh_const_initialized) initialize_llh_const();
    return rllh(x) + llh_const;
}

template<int Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::rllh(const Vec &x) const
{
    return -.5*helpers::compute_quadratic_from_symmetric(num_dim(), (x-mu()).eval(), sigma_inv());
}

template<int Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::grad(const Vec &x) const
{
    return -sigma_inv()*(x-mu()); //Ignore row-vs-col vector ambiguity
}

template<int Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::grad2(const Vec &) const
{
    return -sigma_inv().diag();
}

template<int Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimMatT 
MultivariateNormalDist<Ndim>::hess(const Vec &) const
{
    return -sigma_inv();
}
    
template<int Ndim>
template<class Vec,class Vec2> //Allow different vector types for flexibility between fixed and non-fixed
void MultivariateNormalDist<Ndim>::grad_grad2_accumulate(const Vec &x, Vec2 &g, Vec2 &g2) const
{
    g  += -sigma_inv()*(x-mu());
    g2 += -sigma_inv().diag();
}

template<int Ndim>
template<class Vec,class Vec2,class Mat> //Allow different vector/mat types for flexibility between fixed and non-fixed
void MultivariateNormalDist<Ndim>::grad_hess_accumulate(const Vec &x, Vec2 &g, Mat &hess) const
{
    g += -sigma_inv()*(x-mu());
    hess += -sigma_inv();
}
    
template<int Ndim>
template<class RngT>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::sample(RngT &rng) const
{
    std::normal_distribution<double> unit_normal;
    NdimVecT s;
    for(int i=0;i<Ndim;i++) s(i) = unit_normal(rng);
    return mu()+_sigma_chol*s;
}

template<int Ndim>
double MultivariateNormalDist<Ndim>::compute_llh_const(const NdimMatT &sigma)
{
    double sign;
    double logdet_sigma;
    arma::log_det(logdet_sigma, sign, sigma);
    if(sign<0) throw ParameterValueError("Log determinant is negative.  Sigma is not positive definite.");
    return .5*(logdet_sigma + Ndim*constants::log2pi);
}

template<int Ndim>
void MultivariateNormalDist<Ndim>::initialize_llh_const() const
{
    llh_const = compute_llh_const(sigma());
    llh_const_initialized = true;
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MULTIVARIATENORMALDIST_H */
