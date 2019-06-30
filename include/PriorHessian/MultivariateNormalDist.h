/** @file MultivariateNormalDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
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
template<IdxT Ndim> //, meta::ConstructableIf< (Ndim>=2) > = true>
class MultivariateNormalDist : public MultivariateDist
{
    static constexpr IdxT _num_params= Ndim+(Ndim*Ndim+Ndim)/2;
    
public:
    using NdimVecT = arma::Col<double>::fixed<Ndim>;
    using NdimMatT = arma::Mat<double>::fixed<Ndim,Ndim>;
    using NparamsVecT = arma::Col<double>::fixed<_num_params>;
    
    static constexpr IdxT num_params() {return _num_params;}
    static constexpr IdxT num_dim() {return Ndim;}
    static const NdimVecT& lbound();
    static const NdimVecT& ubound();
    template<class Vec>
    static bool in_bounds(const Vec &u) { return  arma::all(lbound() < u) && arma::all(u < ubound()); }

    static const StringVecT& param_names();
    static const NparamsVecT& param_lbound();
    static const NparamsVecT& param_ubound();
    
    template<class Vec>
    static bool check_mu(const Vec &mu);
    template<class Mat>
    static bool check_sigma(const Mat &sigma);
    template<class Vec, class Mat>
    static bool check_params(const Vec &mu, const Mat &sigma);
    template<class Vec>
    static bool check_params(const Vec &params);

    MultivariateNormalDist();    //Default to unit Gaussian
    template<class Vec, class Mat>
    MultivariateNormalDist(Vec &&mu, Mat &&sigma);

    const NdimVecT& mu() const;
    const NdimMatT& sigma() const;
    const NdimMatT& sigma_inv() const;
    
    template<class Vec> void set_mu(Vec&& val);
    template<class Mat> void set_sigma(Mat&& val);
    
    bool operator==(const MultivariateNormalDist<Ndim> &o) const;    
    bool operator!=(const MultivariateNormalDist<Ndim> &o) const { return !this->operator==(o); }
            
    double get_param(IdxT idx) const;
 
    NparamsVecT params() const;
    template<class Vec>
    void set_params(const Vec &p);

    template<class Vec,class Mat>
    void set_params(Vec &&mu, Mat &&sigma);
    
    /* Import names from Dependent Base Class */    
    NdimVecT mean() const { return mu(); }
    NdimVecT mode() const { return mu(); }
    
    template<class Vec> double cdf(Vec x) const;
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
  
    /* Specialized iterator-based adaptor methods for efficient use by CompositeDist::ComponentDistAdaptor */    
    template<class IterT>
    static bool check_params_iter(IterT &params);

    template<class IterT>
    void append_params(IterT &params) const;

    template<class IterT>
    void set_params_iter(IterT &params);

private:    
    static StringVecT _param_names; //Cannonical names for parameters
    static NparamsVecT _param_lbound; //Lower bound on valid parameter values 
    static NparamsVecT _param_ubound; //Upper bound on valid parameter values
    static NdimVecT _lbound;
    static NdimVecT _ubound;
    
    template<class Vec>
    static NdimMatT compressed_upper_triangular_to_full_matrix(const Vec &v);

    template<class Mat>
    static VecT full_matrix_to_compressed_upper_triangular(const Mat &m);

    static void idx_to_row_col(IdxT idx, IdxT &row, IdxT &col);
    static bool init_param_names();
    static bool init_param_lbound();
    static bool init_param_ubound();
    static bool init_lbound();
    static bool init_ubound();
    
    /* Non-static private members
     */
    NdimVecT _mu;
    NdimMatT _sigma;
    NdimMatT _sigma_inv;
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
template<IdxT Ndim>
StringVecT MultivariateNormalDist<Ndim>::_param_names;

template<IdxT Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::_param_lbound;

template<IdxT Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::_param_ubound;

template<IdxT Ndim>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::_lbound;

template<IdxT Ndim>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::_ubound;
/* Constructors */

template<IdxT Ndim>
MultivariateNormalDist<Ndim>::MultivariateNormalDist() : 
        MultivariateDist(),
        _mu(arma::fill::zeros), 
        _sigma(arma::fill::zeros)
{
    _sigma.diag().ones();
    _sigma_inv = _sigma;//sigma == sigma_inv == eye(Ndim)
    _sigma_chol = _sigma;
    llh_const_initialized = false;
}
    
template<IdxT Ndim>
template<class Vec, class Mat>
MultivariateNormalDist<Ndim>::MultivariateNormalDist(Vec &&mu, Mat &&sigma) 
{
    set_mu(std::forward<Vec>(mu));
    set_sigma(std::forward<Mat>(sigma));
}
    
/* public static methods */
template<IdxT Ndim>
template<class Vec>
bool MultivariateNormalDist<Ndim>::check_mu(const Vec &mu)
{
    return mu.is_finite();
}

template<IdxT Ndim>
template<class Mat>
bool MultivariateNormalDist<Ndim>::check_sigma(const Mat &sigma)
{
    if(!sigma.is_finite()) return false;
    if(arma::any(sigma.diag()<=0)) return false;
    NdimMatT R; 
    return arma::chol(R,arma::symmatu(sigma));//sigma is upper triangular.
}

template<IdxT Ndim>
template<class Vec, class Mat>
bool MultivariateNormalDist<Ndim>::check_params(const Vec &mu, const Mat &sigma)
{
    return check_mu(mu) && check_sigma(sigma);
}

template<IdxT Ndim>
template<class Vec>
bool MultivariateNormalDist<Ndim>::check_params(const Vec &params)
{
    return check_mu(params.head(Ndim)) && 
            check_sigma(compressed_upper_triangular_to_full_matrix(params.tail(num_params()-Ndim)));
}

template<IdxT Ndim>
template<class IterT>
bool MultivariateNormalDist<Ndim>::check_params_iter(IterT &params)
{
    for(IdxT k = 0; k<Ndim; k++) if( !std::isfinite(*params++)) return false;
    NdimMatT S;
    for(IdxT j=0;j<Ndim;j++) for(IdxT i=0;i<=j;i++) S(i,j) = *params++;
    return check_sigma(S);
}


/* private static methods */

template<IdxT Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimMatT 
MultivariateNormalDist<Ndim>::compressed_upper_triangular_to_full_matrix(const Vec &v)
{
    IdxT Z = v.n_elem;
    if(Z != num_params()-Ndim) {
        std::ostringstream msg;
        msg<<"Invalid vector size Z:"<<Z<<" should be:"<<num_params()-Ndim;
        throw ParameterSizeError(msg.str());
    }
    NdimMatT m;
    IdxT n=0;
    for(IdxT j=0;j<Ndim;j++) for(IdxT i=0;i<=j; i++) m(i,j) = v(n++);
    for(IdxT j=0;j<Ndim-1;j++) for(IdxT i=j+1;i<Ndim; i++) m(i,j) = m(j,i); //copy upper triangle to lower.
    return m;
}

template<IdxT Ndim>
template<class Mat>
VecT
MultivariateNormalDist<Ndim>::full_matrix_to_compressed_upper_triangular(const Mat &m)
{
    IdxT Z = num_params()-Ndim;
    VecT v(Z);
    auto it = v.begin();
    for(IdxT j=0;j<Ndim;j++) for(IdxT i=0;i<=j;i++) *it++ = m(i,j);
    return v;
}


template<IdxT Ndim>
const StringVecT& MultivariateNormalDist<Ndim>::param_names()
{ 
    static bool initialized = init_param_names(); //Run initialization only once
    (void) initialized; //Prevent an unused variable warning
    return _param_names; 
}

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NparamsVecT& 
MultivariateNormalDist<Ndim>::param_lbound()
{ 
    static bool initialized = init_param_lbound(); //Run initialization only once
    (void) initialized; //Prevent an unused variable warning
    return _param_lbound;
}

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NparamsVecT& 
MultivariateNormalDist<Ndim>::param_ubound()
{ 
    static bool initialized = init_param_ubound(); //Run initialization only once
    (void) initialized; //Prevent an unused variable warning
    return _param_ubound;
}

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NdimVecT& 
MultivariateNormalDist<Ndim>::lbound()
{ 
    static bool initialized = init_lbound(); //Run initialization only once
    (void) initialized; //Prevent an unused variable warning
    return _lbound;
}

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NdimVecT& 
MultivariateNormalDist<Ndim>::ubound()
{ 
    static bool initialized = init_ubound(); //Run initialization only once
    (void) initialized; //Prevent an unused variable warning
    return _ubound;
}


//translate linear idx into covariance matrix sigma using the ordering of an upper-triangular 
// matrix in col-major form.
template<IdxT Ndim>
void  MultivariateNormalDist<Ndim>::idx_to_row_col(IdxT idx, IdxT &row, IdxT &col)
{
    col = 0;
    IdxT s = 0;
    while(s+col < idx) s+= ++col; //Find the column, keeping truck of the total s of previous elements in previous columns
    row = idx-s;        
}

template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::init_param_names()
{
    _param_names.reserve(num_params());
    for(IdxT k=0;k<Ndim;k++) {
        std::ostringstream name;
        name<<"mu_"<<k+1;
        _param_names.emplace_back(name.str());
    }
    for(IdxT c=0;c<Ndim;c++) for(IdxT r=0;r<=c;r++) {
        std::ostringstream name;
        name<<"sigma_"<<r<<"_"<<c;
        _param_names.emplace_back(name.str());
    }
    return true;
}

template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::init_param_lbound()
{ 
    _param_lbound.fill(-INFINITY);    
    for(IdxT c=1, k=Ndim; k<num_params(); k += ++c) _param_lbound(k)=0; //Diagonal elements of cov are positive
    return true;
}
    
template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::init_param_ubound()
{ 
    _param_ubound.fill(INFINITY);
    return true;
}

template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::init_lbound()
{ 
    _lbound.fill(-INFINITY);
    return true;
}

template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::init_ubound()
{ 
    _ubound.fill(INFINITY);
    return true;
}

/* Non-static methods */

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NdimVecT& 
MultivariateNormalDist<Ndim>::mu() const 
{ return _mu; }

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NdimMatT& 
MultivariateNormalDist<Ndim>::sigma() const 
{ return _sigma; }

template<IdxT Ndim>
const typename MultivariateNormalDist<Ndim>::NdimMatT& 
MultivariateNormalDist<Ndim>::sigma_inv() const 
{ return _sigma_inv; }

// template<IdxT Ndim>
// const typename MultivariateNormalDist<Ndim>::NdimMatT& 
// MultivariateNormalDist<Ndim>::sigma_chol() const 
// { return _sigma_chol; }

template<IdxT Ndim>
template<class Vec>
void MultivariateNormalDist<Ndim>::set_mu(Vec&& val) 
{ if(check_mu(val)) _mu = std::forward<Vec>(val); }

template<IdxT Ndim>
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
        _sigma_inv = arma::symmatu(arma::inv_sympd(arma::symmatu(val)));
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

template<IdxT Ndim>
bool MultivariateNormalDist<Ndim>::operator==(const MultivariateNormalDist<Ndim> &o) const 
{ 
    return arma::all(mu() == o.mu()) && arma::all(arma::all(sigma() == o.sigma())); 
}

template<IdxT Ndim>
typename MultivariateNormalDist<Ndim>::NparamsVecT 
MultivariateNormalDist<Ndim>::params() const
{
    NparamsVecT p;
    p.head(Ndim) = mu();
    p.tail(num_params()-Ndim) = full_matrix_to_compressed_upper_triangular(sigma());
    return p;
}

template<IdxT Ndim>
double MultivariateNormalDist<Ndim>::get_param(IdxT idx) const
{
    if(idx<Ndim) return _mu[idx];
    //otherwise pram is a sigma index as an upper-triangular matrix in col-major form.
    idx -= Ndim;
    IdxT row,col;
    idx_to_row_col(idx,row,col);
    return _sigma(row,col);
}

template<IdxT Ndim>
template<class Vec>
void MultivariateNormalDist<Ndim>::set_params(const Vec &p)
{ 
    set_mu(p.head(Ndim));
    set_sigma(compressed_upper_triangular_to_full_matrix(p.tail(num_params()-Ndim)));
}

template<IdxT Ndim>
template<class Vec,class Mat>
void MultivariateNormalDist<Ndim>::set_params(Vec &&mu_val, Mat &&sigma_val)
{ 
    set_mu(std::forward<Vec>(mu_val));
    set_sigma(std::forward<Mat>(sigma_val));
}

template<IdxT Ndim>
template<class IterT>
void MultivariateNormalDist<Ndim>::set_params_iter(IterT &params)
{
    NdimVecT m;
    std::copy_n(params,Ndim,m.begin());
    params+=Ndim;
    NdimMatT S;
    for(IdxT j=0;j<Ndim;j++) for(IdxT i=0;i<=j;i++) S(i,j) = *params++;
    set_params(std::move(m),std::move(S));
}

template<IdxT Ndim>
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

template<IdxT Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::pdf(const Vec &x) const
{
    // ((2*pi)^Ndim * det(Sigma))^{-1/2} * exp(-1/2*(x-mu)'*Sigma^{-1}*(x-mu))
//     NdimVecT delta = x-mu();
//     return 1./sqrt(arma::det(sigma)) * pow(2*arma::datum::pi,-Ndim/2.) * exp(-.5*delta.t()*arma::inv(sigma)*delta);
    return exp(llh(x));
}

template<IdxT Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::llh(const Vec &x) const
{
    if(!llh_const_initialized) initialize_llh_const();
    return rllh(x) + llh_const;
}

template<IdxT Ndim>
template<class Vec>
double MultivariateNormalDist<Ndim>::rllh(const Vec &x) const
{
    return -.5*helpers::compute_quadratic_from_symmetric(num_dim(), (x-mu()).eval(), sigma_inv());
}

template<IdxT Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::grad(const Vec &x) const
{
    return -sigma_inv()*(x-mu()); //Ignore row-vs-col vector ambiguity
}

template<IdxT Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::grad2(const Vec &) const
{
    return -sigma_inv().diag();
}

template<IdxT Ndim>
template<class Vec>
typename MultivariateNormalDist<Ndim>::NdimMatT 
MultivariateNormalDist<Ndim>::hess(const Vec &) const
{
    return -sigma_inv();
}
    
template<IdxT Ndim>
template<class Vec,class Vec2> //Allow different vector types for flexibility between fixed and non-fixed
void MultivariateNormalDist<Ndim>::grad_grad2_accumulate(const Vec &x, Vec2 &g, Vec2 &g2) const
{
    g  += -sigma_inv()*(x-mu());
    g2 += -sigma_inv().diag();
}

template<IdxT Ndim>
template<class Vec,class Vec2,class Mat> //Allow different vector/mat types for flexibility between fixed and non-fixed
void MultivariateNormalDist<Ndim>::grad_hess_accumulate(const Vec &x, Vec2 &g, Mat &hess) const
{
    g += -sigma_inv()*(x-mu());
    hess += -sigma_inv();
}
    
template<IdxT Ndim>
template<class RngT>
typename MultivariateNormalDist<Ndim>::NdimVecT 
MultivariateNormalDist<Ndim>::sample(RngT &rng) const
{
    std::normal_distribution<double> unit_normal;
    NdimVecT s;
    for(IdxT i=0;i<Ndim;i++) s(i) = unit_normal(rng);
    return mu()+_sigma_chol*s;
}

template<IdxT Ndim>
double MultivariateNormalDist<Ndim>::compute_llh_const(const NdimMatT &sigma)
{
    double sign;
    double logdet_sigma;
    arma::log_det(logdet_sigma, sign, sigma);
    if(sign<0) throw ParameterValueError("Log determinant is negative.  Sigma is not positive definite.");
    return .5*(logdet_sigma + Ndim*constants::log2pi);
}

template<IdxT Ndim>
void MultivariateNormalDist<Ndim>::initialize_llh_const() const
{
    llh_const = compute_llh_const(sigma());
    llh_const_initialized = true;
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MULTIVARIATENORMALDIST_H */
