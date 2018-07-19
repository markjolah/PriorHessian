/** @file test_BoundsAdaptedDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include "test_prior_hessian.h"

using namespace prior_hessian;

                                
TYPED_TEST_CASE(BoundsAdaptedDistTest, UnivariateDistTs);

TYPED_TEST(BoundsAdaptedDistTest, copy_assignment) {
    SCOPED_TRACE("copy_assignment");
    auto &dist = this->dist;
    BoundsAdaptedDistT<TypeParam> dist_copy{};
    dist_copy = dist;
    check_equal(dist, dist_copy);
}

TYPED_TEST(BoundsAdaptedDistTest, copy_construction) {
    SCOPED_TRACE("copy_construction");
    auto &dist = this->dist;
    auto dist_copy{dist};
    check_equal(dist, dist_copy);
}

TYPED_TEST(BoundsAdaptedDistTest, move_assignment) {
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
    //Check bounds are correct
    EXPECT_EQ(dist_copy.lbound(),lbound);
    EXPECT_EQ(dist_copy.ubound(),ubound);
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_EQ(v,v2);
    EXPECT_LE(lbound,v2);
    EXPECT_LE(v2,ubound);
}

TYPED_TEST(BoundsAdaptedDistTest, move_construction) {
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
    //Check bounds are correct
    EXPECT_EQ(dist_copy.lbound(),lbound);
    EXPECT_EQ(dist_copy.ubound(),ubound);
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_EQ(v,v2);
    EXPECT_LE(lbound,v2);
    EXPECT_LE(v2,ubound);
}

TYPED_TEST(BoundsAdaptedDistTest, get_params) {
    auto &dist = this->dist;
    auto params = dist.params();    
    EXPECT_EQ(params.n_elem,dist.num_params());
    for(IdxT i=0; i<dist.num_params(); i++)
        EXPECT_EQ(params[i],dist.get_param(i));
}

TYPED_TEST(BoundsAdaptedDistTest, set_params) {
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

TYPED_TEST(BoundsAdaptedDistTest, get_lbound_ubound) {
    auto &dist = this->dist;
    double lb = dist.lbound();
    EXPECT_FALSE(std::isnan(lb));
    double ub = dist.ubound();
    EXPECT_FALSE(std::isnan(ub));
    EXPECT_LT(lb,ub);
}
    
TYPED_TEST(BoundsAdaptedDistTest, equality_inequality) {
    auto &dist = this->dist;
    auto dist_copy = dist;
    EXPECT_TRUE(dist == dist_copy);
    EXPECT_FALSE(dist != dist_copy);
    dist_copy.set_param(0,dist_copy.get_param(0)+1E-7);
    EXPECT_FALSE(dist == dist_copy);
    EXPECT_TRUE(dist != dist_copy);
}

TYPED_TEST(BoundsAdaptedDistTest, param_names) {
    auto &dist = this->dist;
    auto &params = dist.param_names;
    std::set<std::string> params_set(params.begin(),params.end());
    EXPECT_EQ(params.size(), params_set.size())<< "Parameter Names must be unique.";
    EXPECT_EQ(params.size(), dist.num_params());
    for(IdxT n=0; n<dist.num_params(); n++) EXPECT_FALSE(params[n].empty());
}

TYPED_TEST(BoundsAdaptedDistTest, cdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double cdf = dist.cdf(v);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
    }
}

TYPED_TEST(BoundsAdaptedDistTest, pdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double pdf = dist.pdf(v);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}

TYPED_TEST(BoundsAdaptedDistTest, rllh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double rllh = dist.rllh(v);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}

TYPED_TEST(BoundsAdaptedDistTest, llh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double llh = dist.llh(v);
        EXPECT_TRUE(std::isfinite(llh));
    }
}

TYPED_TEST(BoundsAdaptedDistTest, grad) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double grad = dist.grad(v);
        EXPECT_TRUE(std::isfinite(grad));
    }
}

TYPED_TEST(BoundsAdaptedDistTest, grad2) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        double grad2 = dist.grad2(v);
        EXPECT_TRUE(std::isfinite(grad2));
    }
}

TYPED_TEST(BoundsAdaptedDistTest, grad_grad_accumulate) {
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

TYPED_TEST(BoundsAdaptedDistTest, set_ubound) {
    auto &dist = this->dist;
    auto dist_copy = this->dist;
    double old_bound = dist.ubound();
    double delta = 1.0E0;
    double default_bounds = env->sample_real(dist.icdf(0.55),dist.icdf(0.85));
    double new_bound = std::isfinite(old_bound) ? old_bound + delta : default_bounds;        
    //check set_bounds(,ubound)
    dist.set_bounds(dist.lbound(),new_bound);
    EXPECT_EQ(new_bound,dist.ubound());
    //check set_ubound()
    dist_copy.set_ubound(new_bound);
    EXPECT_EQ(new_bound,dist_copy.ubound());
}

TYPED_TEST(BoundsAdaptedDistTest, set_lbound) {
    auto &dist = this->dist;
    auto dist_copy = this->dist;
    double old_bound = dist.lbound();
    double delta = 1.0E-2;
    double default_bounds = 2;
    double new_bound = std::isfinite(old_bound) ? old_bound + delta : default_bounds;        
    //check set_bounds(,ubound)
    dist.set_bounds(new_bound,dist.ubound());
    EXPECT_EQ(new_bound,dist.lbound());
    //check set_ubound()
    dist_copy.set_lbound(new_bound);
    EXPECT_EQ(new_bound,dist_copy.lbound());
}


TYPED_TEST(BoundsAdaptedDistTest, sample_bounds) {
    auto &dist = this->dist;
    double median = dist.median();
    double lbound = env->sample_real(median/2, median*2);
    double ubound = env->sample_real(lbound+1, lbound*5);
    dist.set_bounds(lbound,ubound);
    for(int n=0; n < this->Ntest; n++){
        double v = dist.sample(env->get_rng());
        EXPECT_LE(lbound,v);
        EXPECT_LE(v,ubound);
    }
}

TYPED_TEST(BoundsAdaptedDistTest, set_illegal_bounds) {
    //check set_bounds()
    auto &dist = this->dist;
    auto dist_copy = dist;
    EXPECT_THROW(dist.set_bounds(1,1), ParameterValueError);
    EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
    EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";
    
    EXPECT_THROW(dist.set_bounds(2,1.3), ParameterValueError);
    EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
    EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";

    double lb = env->sample_real(dist.icdf(0.15), dist.icdf(0.45));
    dist.set_lbound(lb);
    EXPECT_THROW(dist.set_ubound(lb-0.1), ParameterValueError);
    EXPECT_EQ(dist.lbound(), lb)<<"Bounds changed when bad params passsed.";
    EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";
    dist.set_lbound(dist_copy.lbound());
    EXPECT_EQ(dist.lbound(),dist_copy.lbound());
    
    double ub = env->sample_real(dist.icdf(0.55), dist.icdf(0.85));
    dist.set_ubound(ub);
    EXPECT_THROW(dist.set_lbound(ub+0.2), ParameterValueError);
    EXPECT_EQ(dist.ubound(), ub)<<"Bounds changed when bad params passsed.";
    EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
    dist.set_ubound(dist_copy.ubound());
    EXPECT_EQ(dist.ubound(),dist_copy.ubound());
}


TYPED_TEST(BoundsAdaptedDistTest, set_bounds_cdf) {
    auto &dist = this->dist;
    auto dist_copy = dist;
    ASSERT_EQ(dist,dist_copy); //Sanity check
    double median = dist.median();
    double lbound = env->sample_real(median/2, median*2);
    double ubound = env->sample_real(lbound+1, lbound*5);
    dist.set_ubound(ubound);
    EXPECT_EQ(dist.cdf(ubound),1)<<"bad cdf at ubound: "<<ubound;
    if(std::isfinite(dist.lbound())) {
        EXPECT_EQ(dist.cdf(dist.lbound()),0)<<"bad cdf at lbound: "<<dist.lbound();
    }
    dist_copy.set_lbound(lbound);
    if(std::isfinite(dist_copy.ubound())) {
        EXPECT_EQ(dist_copy.cdf(dist_copy.ubound()),1)<<"bad cdf at ubound: "<<dist_copy.ubound();
    }
    EXPECT_EQ(dist_copy.cdf(lbound),0)<<"bad cdf at lbound: "<<lbound;
    dist.set_lbound(lbound);
    dist_copy.set_ubound(ubound);
    EXPECT_EQ(dist,dist_copy);  //Should arrive at same distrbution by either order of bounds setting
}

// TEST_F(NormalDistTest, set_bounds_cdf) {
//     double ubound = 20;
//     dist.set_ubound(ubound);
//     EXPECT_EQ(dist.cdf(ubound),1);
//     double lbound = 0;
//     dist.set_lbound(lbound);
//     EXPECT_EQ(dist.cdf(lbound),0);
// }

