/** @file test_pprior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the ParallelRngManager class
 */
#include "test_prior_hessian.h"

using UnivariateDistTs = ::testing::Types<prior_hessian::NormalDist,
                                          prior_hessian::GammaDist,
                                          prior_hessian::ParetoDist,
                                          prior_hessian::SymmetricBetaDist> ;
TYPED_TEST_CASE(UnivariateDistTest, UnivariateDistTs);


TYPED_TEST(UnivariateDistTest, params) {
    auto params = this->dist.params();    
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    for(size_t i=0; i<this->dist.num_params(); i++)
        EXPECT_EQ(params[i],this->dist.get_param(i));
}

TYPED_TEST(UnivariateDistTest, set_params) {
    auto new_params = this->dist.params();
    for(IdxT k=0; k<new_params.n_elem; k++)
        new_params[k] += 1./double(k+1);
    this->dist.set_params(new_params);
    auto params = this->dist.params();
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    for(IdxT k=0; k<params.n_elem; k++)
        EXPECT_EQ(params[k],new_params[k]);
}

// TYPED_TEST(UnivariateDistTest, params_desc) {
//     auto params = this->dist.params_desc();    
// }

TYPED_TEST(UnivariateDistTest, set_lbound) {
    double new_lbound = std::max(0.5,this->dist.lbound()-0.5);
    //check set_bounds(lbound,)
    this->dist.set_bounds(new_lbound,this->dist.ubound());
    EXPECT_EQ(new_lbound,this->dist.lbound());
    //check set_lbound()
    new_lbound = std::max(0.7,this->dist.lbound()-0.5);
    this->dist.set_lbound(new_lbound);
    EXPECT_EQ(new_lbound,this->dist.lbound());
}

TYPED_TEST(UnivariateDistTest, set_ubound) {
    double new_ubound = this->dist.ubound()+2;
    //check set_bounds(,ubound)
    this->dist.set_bounds(this->dist.lbound(),new_ubound);
    EXPECT_EQ(new_ubound,this->dist.ubound());
    //check set_ubound()
    new_ubound = this->dist.ubound()+3;
    this->dist.set_ubound(new_ubound);
    EXPECT_EQ(new_ubound,this->dist.ubound());
}

// TYPED_TEST(UnivariateDistTest, sample_bounds) {
//     double L=1;
//     double U=10;
//     this->dist.set_bounds(L,U);
//     for(int n=0; n < this->Ntest; n++){
//         double v = this->dist.sample(env->get_rng());
//         EXPECT_LE(L,v);
//         EXPECT_LE(v,U);
//     }
// }

TYPED_TEST(UnivariateDistTest, cdf) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double cdf = this->dist.cdf(v);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
    }
}

TYPED_TEST(UnivariateDistTest, pdf) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double pdf = this->dist.pdf(v);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}

TYPED_TEST(UnivariateDistTest, rllh) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double rllh = this->dist.rllh(v);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}

TYPED_TEST(UnivariateDistTest, llh) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double llh = this->dist.llh(v);
        EXPECT_TRUE(std::isfinite(llh));
    }
}

TYPED_TEST(UnivariateDistTest, grad) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double grad = this->dist.grad(v);
        EXPECT_TRUE(std::isfinite(grad));
    }
}

TYPED_TEST(UnivariateDistTest, grad2) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double grad2 = this->dist.grad2(v);
        EXPECT_TRUE(std::isfinite(grad2));
    }
}

TYPED_TEST(UnivariateDistTest, grad_grad_accumulate) {
    for(int n=0; n < this->Ntest; n++){
        double v = this->dist.sample(env->get_rng());
        double grad = this->dist.grad(v);
        double grad2 = this->dist.grad2(v);
        double grad_acc = 0;
        double grad2_acc = 0;
        this->dist.grad_grad2_accumulate(v,grad_acc,grad2_acc);
        EXPECT_TRUE(std::isfinite(grad_acc));
        EXPECT_TRUE(std::isfinite(grad2_acc));
        EXPECT_DOUBLE_EQ(grad,grad_acc);
        EXPECT_DOUBLE_EQ(grad2,grad2_acc);
    }
}

/*
TEST_F(NormalDistTest, set_ubound) {
    double ubound = this->dist.get_ubound();
    EXPECT_EQ(ubound,INFINITY);
    //check set_bounds()
    double new_ubound = 10;
    this->dist.set_bounds(this->dist.get_lbound(),new_ubound);
    EXPECT_EQ(new_ubound,this->dist.get_ubound());
    //check set_ubound()
    new_ubound = 20;
    this->dist.set_ubound(new_ubound);
    EXPECT_EQ(new_ubound,this->dist.get_ubound());
}

TEST_F(NormalDistTest, set_illegal_bounds) {
    //check set_bounds()
    EXPECT_THROW(this->dist.set_bounds(1,1), prior_hessian::PriorHessianError);
    EXPECT_THROW(this->dist.set_bounds(0,-1), prior_hessian::PriorHessianError);
    this->dist.set_lbound(-1);
    EXPECT_THROW(this->dist.set_ubound(-2), prior_hessian::PriorHessianError);
    this->dist.set_ubound(1);
    EXPECT_THROW(this->dist.set_lbound(2), prior_hessian::PriorHessianError);
}

TEST_F(NormalDistTest, set_bounds_cdf) {
    RngT rng(environment.get_seed());
    std::uniform_real_distribution<double> d(-1e6,1e6);
    double lbound = d(rng);
    d = std::uniform_real_distribution<double>(lbound+std::numeric_limits<double>::epsilon(),1e6);
    double ubound = d(rng);
    this->dist.set_ubound(ubound);
    EXPECT_EQ(this->dist.cdf(ubound),1)<<"ubound: "<<ubound;
    this->dist.set_lbound(lbound);
    EXPECT_EQ(this->dist.cdf(lbound),0)<<"lbound: "<<lbound;
}*/

// TEST_F(NormalDistTest, set_bounds_cdf) {
//     double ubound = 20;
//     this->dist.set_ubound(ubound);
//     EXPECT_EQ(this->dist.cdf(ubound),1);
//     double lbound = 0;
//     this->dist.set_lbound(lbound);
//     EXPECT_EQ(this->dist.cdf(lbound),0);
// }

