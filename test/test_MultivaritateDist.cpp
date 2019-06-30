/** @file test_MultivariaateDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include "test_multivariate.h"

// using MultivariateDistTs = ::testing::Types<
//                                 MultivariateNormalDist<2>,MultivariateNormalDist<4> >;
                                
/* Type parameterized test fixtures */
template<class Dist>
class MultivariateDistTest : public ::testing::Test {
public:    
    Dist dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};

TYPED_TEST_SUITE(MultivariateDistTest, MultivariateDistTs);

TYPED_TEST(MultivariateDistTest, copy_assignment) {
    SCOPED_TRACE("copy_assignment");
    auto &dist = this->dist;
    TypeParam dist_copy{};
    dist_copy = dist;
    check_equal(dist, dist_copy);
}

TYPED_TEST(MultivariateDistTest, copy_construction) {
    SCOPED_TRACE("copy_construction");
    auto &dist = this->dist;
    TypeParam dist_copy(dist);
    check_equal(dist, dist_copy);
}

TYPED_TEST(MultivariateDistTest, move_assignment) {
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
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_TRUE(arma::all(v==v2));
    EXPECT_TRUE(arma::all(lbound<=v2));
    EXPECT_TRUE(arma::all(v2<=ubound));
}

TYPED_TEST(MultivariateDistTest, move_construction) {
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
    //Check repeatability of rng generation
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    EXPECT_TRUE(arma::all(v==v2));
    EXPECT_TRUE(arma::all(lbound<=v2));
    EXPECT_TRUE(arma::all(v2<=ubound));
}

TYPED_TEST(MultivariateDistTest, get_params) {
    auto &dist = this->dist;
    auto params = dist.params();
    EXPECT_EQ(static_cast<const arma::uword>(params.n_elem),dist.num_params());
    for(IdxT i=0; i<dist.num_params(); i++)
        EXPECT_EQ(params[i],dist.get_param(i));
}

TYPED_TEST(MultivariateDistTest, set_params) {
    auto &dist = this->dist;
    auto new_params = dist.params();
    for(IdxT k=0; k<new_params.n_elem; k++)
        new_params[k] += 1./double(k+1);
    dist.set_params(new_params);
    auto params = dist.params();
    EXPECT_EQ(static_cast<const arma::uword>(params.n_elem),dist.num_params());
    for(IdxT k=0; k<params.n_elem; k++)
        EXPECT_EQ(params[k],new_params[k]);
}

TYPED_TEST(MultivariateDistTest, param_lbound_ubound) {
    auto &dist = this->dist;
    auto param_lbound = dist.param_lbound();
    auto param_ubound = dist.param_ubound();
    auto new_params = dist.params();
    EXPECT_TRUE( param_lbound.n_elem == dist.num_params());
    EXPECT_TRUE( param_ubound.n_elem == dist.num_params());
    EXPECT_TRUE( arma::all(param_lbound >= -INFINITY));
    EXPECT_TRUE( arma::all(param_ubound <= INFINITY));
    EXPECT_THROW( dist.set_params(param_lbound), prior_hessian::ParameterValueError);
    EXPECT_TRUE( arma::all(new_params==dist.params())); //check that params is not changed
    EXPECT_THROW( dist.set_params(param_ubound), prior_hessian::ParameterValueError);
    EXPECT_TRUE( arma::all(new_params==dist.params())); //check that params is not changed
   
}

TYPED_TEST(MultivariateDistTest, get_lbound_ubound) {
    auto &dist = this->dist;
    auto lb = dist.lbound();
    EXPECT_FALSE(lb.has_nan());
    auto ub = dist.ubound();
    EXPECT_FALSE(ub.has_nan());
    EXPECT_TRUE(arma::all(lb<ub));
}

TYPED_TEST(MultivariateDistTest, equality_inequality) {
    auto &dist = this->dist;
    auto dist_copy = dist;
    EXPECT_TRUE(dist == dist_copy);
    EXPECT_FALSE(dist != dist_copy);
    VecT new_params = dist_copy.params()+1E-7;
    
    dist_copy.set_params(new_params);
    EXPECT_FALSE(dist == dist_copy);
    EXPECT_TRUE(dist != dist_copy);
}

TYPED_TEST(MultivariateDistTest, param_names) {
    auto &dist = this->dist;
    auto params = dist.param_names();
    std::set<std::string> params_set(params.begin(),params.end());
    EXPECT_EQ(params.size(), params_set.size())<< "Parameter Names must be unique.";
    EXPECT_EQ(params.size(), dist.num_params());
    for(IdxT n=0; n<dist.num_params(); n++) EXPECT_FALSE(params[n].empty());
}

TYPED_TEST(MultivariateDistTest, cdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double cdf = dist.cdf(v);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
        EXPECT_LE(cdf,1);
    }
}

TYPED_TEST(MultivariateDistTest, pdf) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double pdf = dist.pdf(v);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}

TYPED_TEST(MultivariateDistTest, rllh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double rllh = dist.rllh(v);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}

TYPED_TEST(MultivariateDistTest, llh) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        double llh = dist.llh(v);
        EXPECT_TRUE(std::isfinite(llh));
    }
}

TYPED_TEST(MultivariateDistTest, rllh_constant) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v1 = dist.sample(env->get_rng());
        auto v2 = dist.sample(env->get_rng());
        double rllh1 = dist.rllh(v1);
        double rllh2 = dist.rllh(v2);
        double llh1 = dist.llh(v1);
        double llh2 = dist.llh(v2);
        EXPECT_TRUE(std::isfinite(rllh1));
        EXPECT_TRUE(std::isfinite(rllh2));
        EXPECT_TRUE(std::isfinite(llh1));
        EXPECT_TRUE(std::isfinite(llh2));
        EXPECT_DOUBLE_EQ( llh1 - rllh1, llh2-rllh2);
    }
}

TYPED_TEST(MultivariateDistTest, grad) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad = dist.grad(v);
        EXPECT_TRUE(grad.n_elem == dist.num_dim());
        EXPECT_TRUE(grad.is_finite());
    }
}

TYPED_TEST(MultivariateDistTest, grad2) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        auto grad2 = dist.grad2(v);
        EXPECT_TRUE(grad2.n_elem == dist.num_dim());
        EXPECT_TRUE(grad2.is_finite());
    }
}

TYPED_TEST(MultivariateDistTest, hess) {
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

TYPED_TEST(MultivariateDistTest, grad_grad2_accumulate) {
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

TYPED_TEST(MultivariateDistTest, grad_hess_accumulate) {
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

TYPED_TEST(MultivariateDistTest, sample_in_bounds) {
    auto &dist = this->dist;
    for(int n=0; n < this->Ntest; n++){
        auto v = dist.sample(env->get_rng());
        EXPECT_TRUE(v.is_finite());
        EXPECT_TRUE(dist.in_bounds(v));
    }
}
