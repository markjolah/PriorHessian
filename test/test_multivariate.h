/** @file test_multivariate.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief 
 */

#ifndef TEST_MULTIVARIATE_H
#define TEST_MULTIVARIATE_H

#include "test_prior_hessian.h"
#include "PriorHessian/TruncatedMultivariateNormalDist.h"

using namespace prior_hessian;


template<class Dist>
void initialize_dist(meta::ReturnIfSubclassT<Dist,Dist,MultivariateDist> &d)
{
    //hyper-params
    int N = d.num_dim();
    double mu_mean = 0.0;
    double mu_sigma = 1.0;
    double sigma_scale = 2.0;
    double sigma_shape = 2.0;

    d.set_mu(env->sample_normal_vec(N,mu_mean,mu_sigma));
    d.set_sigma(env->sample_sigma_mat(env->sample_gamma_vec(N,sigma_scale,sigma_shape)));
}

template<class Dist>
meta::ReturnIfSubclassT<Dist,Dist,MultivariateDist>
make_dist()
{
    Dist dist;
    initialize_dist<Dist>(dist);
    return dist;
}


template<class Dist1, class Dist2>
meta::ReturnIfSubclassT<void, 
    meta::ReturnIfSubclassT<Dist1, Dist2, MultivariateDist>,MultivariateDist>
check_equal(const Dist1 &d1, const Dist2 &d2)
{
    auto Nparams = d1.num_params();
    ASSERT_EQ(d1,d2);
    ASSERT_EQ(Nparams, d2.num_params());
    //parameters are equal
    auto params1 = d1.params();
    auto params2 = d2.params();
    EXPECT_TRUE(arma::all(params1 == params2));
    EXPECT_EQ(d1,d2);    
    //Check repeatability of rng generation
    env->reset_rng();
    auto v1 = d1.sample(env->get_rng());
    env->reset_rng();
    auto v2 = d2.sample(env->get_rng());
    
    EXPECT_TRUE(arma::all(v1==v2));
}

using MultivariateDistTs = ::testing::Types< MultivariateNormalDist<2>,
                                             MultivariateNormalDist<4> >;

#endif /* TEST_MULTIVARIATE_H */

