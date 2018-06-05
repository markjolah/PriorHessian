/** @file test_pprior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the ParallelRngManager class
 */
#include "test_prior_hessian.h"

using UnivariateDistTs = ::testing::Types<prior_hessian::NormalDist> ;
TYPED_TEST_CASE(UnivariateDistTest, UnivariateDistTs);


TYPED_TEST(UnivariateDistTest, get_params) {
    auto params = this->dist.get_params();    
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    for(size_t i=0; i<this->dist.num_params(); i++)
        EXPECT_EQ(params[i],this->dist.get_param(i));
}
/*
TEST_F(NormalDistTest, set_params) {
    auto new_params = this->dist.get_params();
    new_params[0]+=2;
    new_params[1]+=5;
    this->dist.set_params(new_params);
    auto params = this->dist.get_params();
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    EXPECT_EQ(params[0],new_params[0]);
    EXPECT_EQ(params[1],new_params[1]);
}

TEST_F(NormalDistTest, set_lbound) {
    double lbound = this->dist.get_lbound();
    EXPECT_EQ(lbound,-INFINITY);
    //check set_bounds()
    double new_lbound = -10;
    this->dist.set_bounds(new_lbound,this->dist.get_ubound());
    EXPECT_EQ(new_lbound,this->dist.get_lbound());
    //check set_lbound()
    new_lbound = -20;
    this->dist.set_lbound(new_lbound);
    EXPECT_EQ(new_lbound,this->dist.get_lbound());
}

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

