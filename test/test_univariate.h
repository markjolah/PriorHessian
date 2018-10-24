/** @file test_univariate.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Common include for all testing modules
 */
#include "test_prior_hessian.h"
#include "PriorHessian/TruncatedNormalDist.h"
#include "PriorHessian/TruncatedGammaDist.h"
#include "PriorHessian/TruncatedParetoDist.h"
#include "PriorHessian/ScaledSymmetricBetaDist.h"

using namespace prior_hessian;

template<class Dist>
void initialize_dist(meta::DerivedFrom<Dist,NormalDist> &d)
{
    //hyper-params
    double mu_mean = 3.0;
    double mu_sigma = 7.0;
    double sigma_scale = 3.0;
    double sigma_shape = 2.0;
    d.set_param(0,env->sample_normal(mu_mean,mu_sigma));
    d.set_param(1,env->sample_gamma(sigma_scale,sigma_shape));
}

template<class Dist>
void initialize_dist(meta::DerivedFrom<Dist,GammaDist> &d)
{
    //hyper-params
    double scale_scale = 10.0;
    double scale_shape = 1.5;
    double shape_scale = 10.0;
    double shape_shape = 1.5;
    d.set_param(0,env->sample_gamma(scale_scale,scale_shape));
    d.set_param(1,env->sample_gamma(shape_scale,shape_shape));
}

template<class Dist>
void initialize_dist(meta::DerivedFrom<Dist,ParetoDist> &d)
{
    //hyper-params
    double alpha_scale = 2.0;
    double alpha_shape = 2.0;
    double lbound_scale = 10.0;
    double lbound_shape = 1.5;
    d.set_param(0,env->sample_gamma(alpha_scale,alpha_shape));
    d.set_lbound(env->sample_gamma(lbound_scale,lbound_shape));
}

template<class Dist>
void initialize_dist(meta::DerivedFrom<Dist,SymmetricBetaDist> &d)
{
    //hyper-params
    double beta_scale = 10.0;
    double beta_shape = 1.5;
    d.set_param(0,env->sample_gamma(beta_scale,beta_shape));
}

/* Factory functions */
template<class Dist> 
Dist make_dist()
{
    Dist dist;
    initialize_dist<Dist>(dist);
    return dist;
}

template<class Dist>
meta::EnableIfSubclassT<Dist,UnivariateDist>
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
    
    EXPECT_EQ(v1,v2);
}

using UnivariateDistTs = ::testing::Types<
                                NormalDist, TruncatedNormalDist,
                                GammaDist, TruncatedGammaDist,
                                ParetoDist, TruncatedParetoDist,
                                SymmetricBetaDist, ScaledSymmetricBetaDist>;
                                        

/* Type parameterized test fixtures */
template<class Dist>
class UnivariateDistTest : public ::testing::Test {
public:    
    Dist dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};
