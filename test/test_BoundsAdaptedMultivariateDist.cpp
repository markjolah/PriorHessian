/** @file test_BoundsAdaptedMultivariatDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include "test_multivariate.h"

using namespace prior_hessian;

template<class Dist>
class BoundsAdaptedMultivariateDistTest: public ::testing::Test {
public:
    Dist orig_dist;
    using BoundedDistT = BoundsAdaptedDistT<Dist>;
    BoundedDistT dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        orig_dist = make_dist<Dist>();
        dist = make_adapted_bounded_dist(orig_dist);
    }
};

                                
TYPED_TEST_CASE(BoundsAdaptedMultivariateDistTest, MultivariateDistTs);

TYPED_TEST(BoundsAdaptedMultivariateDistTest, copy_assignment) {
    SCOPED_TRACE("copy_assignment");
    auto &dist = this->dist;
    BoundsAdaptedDistT<TypeParam> dist_copy{};
    dist_copy = dist;
    check_equal(dist, dist_copy);
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, copy_construction) {
    SCOPED_TRACE("copy_construction");
    auto &dist = this->dist;
    auto dist_copy{dist};
    check_equal(dist, dist_copy);
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, move_assignment) {
    auto &dist = this->dist;
    TypeParam dist_copy{};
    volatile double foo_ = (dist_copy.get_param(0),0); //Force something to happen with dist_copy first
    (void) foo_;
    auto params = dist.params();
    IdxT Nparams = dist.num_params();
    auto lbound = dist.lbound();
    auto ubound = dist.ubound();
    env->reset_rng();
    auto v = dist.sample(env->get_rng());

    dist_copy = std::move(dist);  //Now do the move assignment

    //Check copy of parameters is successful
    for(IdxT i=0; i<Nparams; i++)
        EXPECT_EQ(dist_copy.get_param(i),params[i]);
    //Check bounds are correct
    EXPECT_TRUE(arma::all(dist_copy.lbound()==lbound));
    EXPECT_TRUE(arma::all(dist_copy.ubound()==ubound));
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_TRUE(arma::all(v==v2));
    EXPECT_TRUE(arma::all(lbound<v2));
    EXPECT_TRUE(arma::all(v2<ubound));
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, move_construction) {
    auto &dist = this->dist;
    auto params = dist.params();
    IdxT Nparams = dist.num_params();
    auto lbound = dist.lbound();
    auto ubound = dist.ubound();
    env->reset_rng();
    auto v = dist.sample(env->get_rng());

    TypeParam dist_copy = std::move(dist);  //Now do the move construct

    //Check copy of parameters is successful
    for(IdxT i=0; i<Nparams; i++)
        EXPECT_EQ(dist_copy.get_param(i),params[i]);
    //Check bounds are correct
    EXPECT_TRUE(arma::all(dist_copy.lbound()==lbound));
    EXPECT_TRUE(arma::all(dist_copy.ubound()==ubound));
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_TRUE(arma::all(v==v2));
    EXPECT_TRUE(arma::all(lbound<v2));
    EXPECT_TRUE(arma::all(v2<ubound));
}


TYPED_TEST(BoundsAdaptedMultivariateDistTest, get_lbound_ubound) {
    auto &dist = this->dist;
    auto lb = dist.lbound();
    EXPECT_FALSE(lb.has_nan());
    EXPECT_TRUE(arma::all(lb>=dist.global_lbound()));
    auto ub = dist.ubound();
    EXPECT_FALSE(ub.has_nan());
    EXPECT_TRUE(arma::all(ub<=dist.global_ubound()));

    EXPECT_TRUE(arma::all(lb<ub));
}
    
TYPED_TEST(BoundsAdaptedMultivariateDistTest, equality_inequality) {
    auto &dist = this->dist;
    auto dist_copy = dist;
    EXPECT_TRUE(dist == dist_copy);
    EXPECT_FALSE(dist != dist_copy);
    auto params = dist_copy.params();
    params(0)+=1E-7;
    dist_copy.set_params(params);
    EXPECT_FALSE(dist == dist_copy);
    EXPECT_TRUE(dist != dist_copy);
}


TYPED_TEST(BoundsAdaptedMultivariateDistTest, cdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double cdf = dist.cdf(v);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
        EXPECT_LE(cdf,1);
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, pdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double pdf = dist.pdf(v);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, rllh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double rllh = dist.rllh(v);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, llh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double llh = dist.llh(v);
        EXPECT_TRUE(std::isfinite(llh));
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, grad) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad = dist.grad(v);
        EXPECT_TRUE(grad.is_finite());
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, grad2) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad2 = dist.grad2(v);
        EXPECT_TRUE(grad2.is_finite());
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, hess) {
    SCOPED_TRACE("hess");
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto hess = dist.hess(v);
        EXPECT_TRUE(hess.n_rows == dist.num_dim());
        EXPECT_TRUE(hess.n_cols == dist.num_dim());
        EXPECT_TRUE(hess.is_finite());
        check_symmetric(hess);
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, grad_grad2_accumulate) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad = dist.grad(v);
        auto grad2 = dist.grad2(v);
        EXPECT_TRUE(grad.n_elem == dist.num_dim());
        EXPECT_TRUE(grad2.n_elem == dist.num_dim());
        EXPECT_TRUE(grad.is_finite());
        EXPECT_TRUE(grad2.is_finite());
        //grad and grad2 are ok.  Check accumulated versions
        VecT grad_acc(arma::size(grad),arma::fill::zeros);
        VecT grad2_acc(arma::size(grad2),arma::fill::zeros);
        dist.grad_grad2_accumulate(v,grad_acc,grad2_acc);
        EXPECT_TRUE(grad_acc.is_finite());
        EXPECT_TRUE(grad2_acc.is_finite());
        EXPECT_TRUE(arma::approx_equal(grad,grad_acc,"reldiff",1E-9));
        EXPECT_TRUE(arma::approx_equal(grad2,grad2_acc,"reldiff",1E-9));
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, grad_hess_accumulate) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad = dist.grad(v);
        auto hess = dist.hess(v);
        EXPECT_TRUE(grad.n_elem == dist.num_dim());
        EXPECT_TRUE(grad.is_finite());
        EXPECT_TRUE(hess.n_rows == dist.num_dim());
        EXPECT_TRUE(hess.n_cols == dist.num_dim());
        EXPECT_TRUE(hess.is_finite());
        check_symmetric(hess);

        //grad and hess are ok.  Check accumulated versions
        VecT grad_acc(arma::size(grad),arma::fill::zeros);
        MatT hess_acc(arma::size(hess),arma::fill::zeros);
        dist.grad_hess_accumulate(v,grad_acc,hess_acc);
        EXPECT_TRUE(grad_acc.is_finite());
        EXPECT_TRUE(hess_acc.is_finite());
        EXPECT_TRUE(arma::approx_equal(grad,grad_acc,"reldiff",1E-9));
        EXPECT_TRUE(arma::approx_equal(hess,hess_acc,"reldiff",1E-9));
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, sample_in_bounds) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        EXPECT_TRUE(v.is_finite());
        EXPECT_TRUE(dist.in_bounds(v));
    }
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, set_ubound) {
    auto &dist = this->dist;
    auto dist_copy = this->dist;
    MatT samples(dist.num_dim(),this->Ntest);
    for(int n=0; n < this->Ntest; n++){
        samples.col(n) = dist.sample(env->get_rng());
    }
    VecT new_bound = max(samples,1);
    VecT old_bound = dist.ubound();
    //check set_bounds(,ubound)
    dist.set_bounds(dist.lbound(),new_bound);
    EXPECT_TRUE(arma::all(new_bound==dist.ubound()));
    //check set_ubound()
    dist_copy.set_ubound(old_bound);
    EXPECT_TRUE(arma::all(old_bound==dist_copy.ubound()));
}

TYPED_TEST(BoundsAdaptedMultivariateDistTest, set_lbound) {
    auto &dist = this->dist;
    auto dist_copy = this->dist;
    MatT samples(dist.num_dim(),this->Ntest);
    for(int n=0; n < this->Ntest; n++){
        samples.col(n) = dist.sample(env->get_rng());
    }
    VecT new_bound = min(samples,1);
    VecT old_bound = dist.lbound();
    //check set_bounds(,ubound)
    dist.set_bounds(new_bound, dist.ubound());
    EXPECT_TRUE(arma::all(new_bound==dist.lbound()));
    //check set_ubound()
    dist_copy.set_lbound(old_bound);
    EXPECT_TRUE(arma::all(old_bound==dist_copy.lbound()));
}


// 
// TYPED_TEST(BoundsAdaptedMultivariateDistTest, set_lbound) {
//     auto &dist = this->dist;
//     auto dist_copy = this->dist;
//     double new_bound = env->sample_real(dist.icdf(0.05),dist.icdf(0.45));
//     //check set_bounds(,ubound)
//     dist.set_bounds(new_bound,dist.ubound());
//     EXPECT_EQ(new_bound,dist.lbound());
//     //check set_ubound()
//     dist_copy.set_lbound(new_bound);
//     EXPECT_EQ(new_bound,dist_copy.lbound());
// }
// 
// 
// TYPED_TEST(BoundsAdaptedMultivariateDistTest, sample_bounds) {
//     auto &dist = this->dist;
//     double lbound = env->sample_real(dist.icdf(0.01), dist.icdf(0.499));
//     double ubound = env->sample_real(dist.icdf(0.501), dist.icdf(0.999));
//     dist.set_bounds(lbound,ubound);
//     for(int n=0; n < this->Ntest; n++){
//         double v = dist.sample(env->get_rng());
//         EXPECT_LE(lbound,v);
//         EXPECT_LE(v,ubound);
//     }
// }
// 
// TYPED_TEST(BoundsAdaptedMultivariateDistTest, set_illegal_bounds) {
//     //check set_bounds()
//     auto &dist = this->dist;
//     auto dist_copy = dist;
//     EXPECT_THROW(dist.set_bounds(1,1), ParameterValueError);
//     EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
//     EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";
//     
//     EXPECT_THROW(dist.set_bounds(2,1.3), ParameterValueError);
//     EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
//     EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";
// 
//     double lb = env->sample_real(dist.icdf(0.15), dist.icdf(0.45));
//     dist.set_lbound(lb);
//     EXPECT_THROW(dist.set_ubound(lb-0.1), ParameterValueError);
//     EXPECT_EQ(dist.lbound(), lb)<<"Bounds changed when bad params passsed.";
//     EXPECT_EQ(dist.ubound(), dist_copy.ubound())<<"Bounds changed when bad params passsed.";
//     dist.set_lbound(dist_copy.lbound());
//     EXPECT_EQ(dist.lbound(),dist_copy.lbound());
//     
//     double ub = env->sample_real(dist.icdf(0.55), dist.icdf(0.85));
//     dist.set_ubound(ub);
//     EXPECT_THROW(dist.set_lbound(ub+0.2), ParameterValueError);
//     EXPECT_EQ(dist.ubound(), ub)<<"Bounds changed when bad params passsed.";
//     EXPECT_EQ(dist.lbound(), dist_copy.lbound())<<"Bounds changed when bad params passsed.";
//     dist.set_ubound(dist_copy.ubound());
//     EXPECT_EQ(dist.ubound(),dist_copy.ubound());
// }
// 
// 
// TYPED_TEST(BoundsAdaptedMultivariateDistTest, set_bounds_cdf) {
//     auto &dist = this->dist;
//     auto dist_copy = dist;
//     ASSERT_EQ(dist,dist_copy); //Sanity check
//     double lbound = env->sample_real(dist.icdf(0.01), dist.icdf(0.45));
//     double ubound = env->sample_real(dist.icdf(0.51), dist.icdf(0.999));
//     dist.set_ubound(ubound);
//     EXPECT_EQ(dist.cdf(ubound),1)<<"bad cdf at ubound: "<<ubound;
//     if(std::isfinite(dist.lbound())) {
//         EXPECT_EQ(dist.cdf(dist.lbound()),0)<<"bad cdf at lbound: "<<dist.lbound();
//     }
//     dist_copy.set_lbound(lbound);
//     if(std::isfinite(dist_copy.ubound())) {
//         EXPECT_EQ(dist_copy.cdf(dist_copy.ubound()),1)<<"bad cdf at ubound: "<<dist_copy.ubound();
//     }
//     EXPECT_EQ(dist_copy.cdf(lbound),0)<<"bad cdf at lbound: "<<lbound;
//     dist.set_lbound(lbound);
//     dist_copy.set_ubound(ubound);
//     EXPECT_EQ(dist,dist_copy);  //Should arrive at same distrbution by either order of bounds setting
// }


