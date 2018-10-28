/** @file test_multivariate.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief 
 */
#include "test_prior_hessian.h"
#include "PriorHessian/TruncatedMultivariateNormalDist.h"

using namespace prior_hessian;

template<class Dist>
void initialize_dist(meta::EnableIfInstantiatedFromNumericT<Dist,MultivariateNormalDist> &d)
{
    //hyper-params
    int N = d.num_dim();
    double mu_mean = -3.0;
    double mu_sigma = 17.0;
    double sigma_scale = 3.0;
    double sigma_shape = 2.0;
    VecT mean(N);
    MatT R(N,N);
    VecT sigma(N);
    for(int i=0; i<N; i++) {
        mean(i) = env->sample_normal(mu_mean,mu_sigma);
        sigma(i) = env->sample_gamma(sigma_scale,sigma_shape);
    }
    //Generate a random orthonormal matrix from a random matrix of standard normal variants.
    for(int i=0; i<N; i++) for(int j=0; j<N; j++) R(j,i) = env->sample_normal(0,1);
    MatT P = arma::orth(R);
    MatT cov = P.t()*arma::diagmat(sigma)*P;
//     std::cout<<"cov: "<<cov;
//     auto ev = arma::eig_sym(cov);
//     std::cout<<"eig_vals:"<<ev.t();
//     auto _chol_u = arma::chol(cov,"upper");
//     auto _chol_l = arma::chol(cov,"lower");
//     std::cout<<"chol_u:"<<_chol_u;
//     std::cout<<"chol_l:"<<_chol_l;
    d.set_mu(mean);
    d.set_sigma(cov);
}

template<class Dist> 
Dist make_dist()
{
    Dist dist;
    initialize_dist<Dist>(dist);
    return dist;
}


template<class Dist>
meta::EnableIfSubclassOfNumericTemplateT<Dist,MultivariateDist>
check_equal(const Dist &d1, const Dist &d2)
{
    auto Nparams = d1.num_params();
    ASSERT_EQ(d1,d2);
    ASSERT_EQ(Nparams, d2.num_params());
    //parameters are equal
    for(IdxT i=0; i<Nparams; i++) EXPECT_EQ(d1.get_param(i),d2.get_param(i));    
    //Check repeatability of rng generation
    env->reset_rng();
    auto v1 = d1.sample(env->get_rng());
    env->reset_rng();
    auto v2 = d2.sample(env->get_rng());
    
    EXPECT_TRUE(arma::all(v1==v2));
}

                                        

using MultivariateDistTs = ::testing::Types< MultivariateNormalDist<2>,
                                             MultivariateNormalDist<4> >;

