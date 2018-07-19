/** @file test_pprior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the ParallelRngManager class
 */
#include "test_prior_hessian.h"

using namespace prior_hessian;

using UnivariateDistTs = ::testing::Types<
                                NormalDist, BoundedNormalDist,
                                GammaDist, BoundedGammaDist,
                                ParetoDist, BoundedParetoDist,
                                SymmetricBetaDist, ScaledSymmetricBetaDist>;
                                        
TYPED_TEST_CASE(UnivariateDistTest, UnivariateDistTs);

template<class Dist>
void check_univariate_dists_equal(const Dist &d1, const Dist &d2)
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

TYPED_TEST(UnivariateDistTest, copy_assignment) {
    SCOPED_TRACE("copy_assignment");
    auto &dist = this->dist;
    TypeParam dist_copy{};
    dist_copy = dist;
    check_univariate_dists_equal(dist, dist_copy);
}

TYPED_TEST(UnivariateDistTest, copy_construction) {
    SCOPED_TRACE("copy_construction");
    auto &dist = this->dist;
    TypeParam dist_copy(dist);
    check_univariate_dists_equal(dist, dist_copy);
}

TYPED_TEST(UnivariateDistTest, move_assignment) {
    auto &dist = this->dist;
    TypeParam dist_copy{};
    volatile double foo_ = (dist_copy.get_param(0),0); //Force something to happen with dist_copy first
    (void) foo_;
    auto params = dist.params();
    IdxT Nparams = dist.num_params();
    double lbound = dist.lbound();
    double ubound = dist.ubound();
    env->reset_rng();
    auto v = dist.sample(env->get_rng());

    dist_copy = std::move(dist);  //Now do the move assignment

    //Check copy of parameters is successful
    for(IdxT i=0; i<Nparams; i++)
        EXPECT_EQ(dist_copy.get_param(i),params[i]);
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_EQ(v,v2);
    EXPECT_LE(lbound,v2);
    EXPECT_LE(v2,ubound);
}

TYPED_TEST(UnivariateDistTest, move_construction) {
    auto &dist = this->dist;
    auto params = dist.params();
    IdxT Nparams = dist.num_params();
    double lbound = dist.lbound();
    double ubound = dist.ubound();
    env->reset_rng();
    auto v = dist.sample(env->get_rng());

    TypeParam dist_copy = std::move(dist);  //Now do the move construct

    //Check copy of parameters is successful
    for(IdxT i=0; i<Nparams; i++)
        EXPECT_EQ(dist_copy.get_param(i),params[i]);
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_EQ(v,v2);
    EXPECT_LE(lbound,v2);
    EXPECT_LE(v2,ubound);
}


TYPED_TEST(UnivariateDistTest, params) {
    auto &dist = this->dist;
    auto params = dist.params();    
    EXPECT_EQ(params.n_elem,dist.num_params());
    for(IdxT i=0; i<dist.num_params(); i++)
        EXPECT_EQ(params[i],dist.get_param(i));
}

TYPED_TEST(UnivariateDistTest, set_params) {
    auto &dist = this->dist;
    auto new_params = dist.params();
    for(IdxT k=0; k<new_params.n_elem; k++)
        new_params[k] += 1./double(k+1);
    dist.set_params(new_params);
    auto params = dist.params();
    EXPECT_EQ(params.n_elem,dist.num_params());
    for(IdxT k=0; k<params.n_elem; k++)
        EXPECT_EQ(params[k],new_params[k]);
}

// TYPED_TEST(UnivariateDistTest, equality) {



// TYPED_TEST(UnivariateDistTest, param_names) {
//     auto params = dist.param_names();    
// }

// TYPED_TEST(UnivariateDistTest, set_lbound) {
//     double new_lbound = dist.lbound()+0.01;
//     //check set_bounds(lbound,)
//     dist.set_bounds(new_lbound,dist.ubound());
//     EXPECT_EQ(new_lbound,dist.lbound());
//     //check set_lbound()
//     new_lbound = dist.lbound()+0.002;
//     dist.set_lbound(new_lbound);
//     EXPECT_EQ(new_lbound,dist.lbound());
// }
// 
// TYPED_TEST(UnivariateDistTest, set_ubound) {
//     double new_ubound = dist.ubound()+0.003;
//     //check set_bounds(,ubound)
//     dist.set_bounds(dist.lbound(),new_ubound);
//     EXPECT_EQ(new_ubound,dist.ubound());
//     //check set_ubound()
//     new_ubound = dist.ubound()+0.003;
//     dist.set_ubound(new_ubound);
//     EXPECT_EQ(new_ubound,dist.ubound());
// }

// TYPED_TEST(UnivariateDistTest, sample_bounds) {
//     double L=1;
//     double U=10;
//     dist.set_bounds(L,U);
//     for(int n=0; n < this->Ntest; n++){
//         double v = dist.sample(env->get_rng());
//         EXPECT_LE(L,v);
//         EXPECT_LE(v,U);
//     }
// }

TYPED_TEST(UnivariateDistTest, cdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double cdf = dist.cdf(v);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
    }
}

TYPED_TEST(UnivariateDistTest, pdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double pdf = dist.pdf(v);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}

TYPED_TEST(UnivariateDistTest, rllh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double rllh = dist.rllh(v);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}

TYPED_TEST(UnivariateDistTest, llh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double llh = dist.llh(v);
        EXPECT_TRUE(std::isfinite(llh));
    }
}

TYPED_TEST(UnivariateDistTest, grad) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double grad = dist.grad(v);
        EXPECT_TRUE(std::isfinite(grad));
    }
}

TYPED_TEST(UnivariateDistTest, grad2) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double grad2 = dist.grad2(v);
        EXPECT_TRUE(std::isfinite(grad2));
    }
}

TYPED_TEST(UnivariateDistTest, grad_grad_accumulate) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double grad = dist.grad(v);
        double grad2 = dist.grad2(v);
        double grad_acc = 0;
        double grad2_acc = 0;
        dist.grad_grad2_accumulate(v,grad_acc,grad2_acc);
        EXPECT_TRUE(std::isfinite(grad_acc));
        EXPECT_TRUE(std::isfinite(grad2_acc));
        EXPECT_DOUBLE_EQ(grad,grad_acc);
        EXPECT_DOUBLE_EQ(grad2,grad2_acc);
    }
}

/*
TEST_F(NormalDistTest, set_ubound) {
    double ubound = dist.get_ubound();
    EXPECT_EQ(ubound,INFINITY);
    //check set_bounds()
    double new_ubound = 10;
    dist.set_bounds(dist.get_lbound(),new_ubound);
    EXPECT_EQ(new_ubound,dist.get_ubound());
    //check set_ubound()
    new_ubound = 20;
    dist.set_ubound(new_ubound);
    EXPECT_EQ(new_ubound,dist.get_ubound());
}

TEST_F(NormalDistTest, set_illegal_bounds) {
    //check set_bounds()
    EXPECT_THROW(dist.set_bounds(1,1), prior_hessian::PriorHessianError);
    EXPECT_THROW(dist.set_bounds(0,-1), prior_hessian::PriorHessianError);
    dist.set_lbound(-1);
    EXPECT_THROW(dist.set_ubound(-2), prior_hessian::PriorHessianError);
    dist.set_ubound(1);
    EXPECT_THROW(dist.set_lbound(2), prior_hessian::PriorHessianError);
}

TEST_F(NormalDistTest, set_bounds_cdf) {
    RngT rng(environment.get_seed());
    std::uniform_real_distribution<double> d(-1e6,1e6);
    double lbound = d(rng);
    d = std::uniform_real_distribution<double>(lbound+std::numeric_limits<double>::epsilon(),1e6);
    double ubound = d(rng);
    dist.set_ubound(ubound);
    EXPECT_EQ(dist.cdf(ubound),1)<<"ubound: "<<ubound;
    dist.set_lbound(lbound);
    EXPECT_EQ(dist.cdf(lbound),0)<<"lbound: "<<lbound;
}*/

// TEST_F(NormalDistTest, set_bounds_cdf) {
//     double ubound = 20;
//     dist.set_ubound(ubound);
//     EXPECT_EQ(dist.cdf(ubound),1);
//     double lbound = 0;
//     dist.set_lbound(lbound);
//     EXPECT_EQ(dist.cdf(lbound),0);
// }

