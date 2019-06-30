/** @file test_MultivariateNormalDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include "test_multivariate.h"

using MultivariateNormalDistTs = ::testing::Types<
                                MultivariateNormalDist<2>,MultivariateNormalDist<4> >;

/* Type parameterized test fixtures */
template<class Dist>
class MultivariateNormalDistTest : public ::testing::Test {
public:    
    Dist dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};

TYPED_TEST_SUITE_COMPAT(MultivariateNormalDistTest, MultivariateNormalDistTs);

TYPED_TEST(MultivariateNormalDistTest, set_mu) {
    auto &dist = this->dist;
    auto new_mu = dist.mu();
    for(IdxT k=0; k<new_mu.n_elem; k++) new_mu[k] += 1./double(k+1);
    dist.set_mu(new_mu);
    auto mu = dist.mu();
    EXPECT_EQ(static_cast<const arma::uword>(mu.n_elem),dist.num_dim());
    EXPECT_TRUE(arma::all(mu==new_mu));
}

TYPED_TEST(MultivariateNormalDistTest, set_sigma) {
    auto &dist = this->dist;
    auto new_sigma = dist.sigma();
    for(IdxT j=0; j<new_sigma.n_cols; j++) for(IdxT i=0; i<=j; i++) new_sigma(i,j) += 1./double(i+3.141*j+1);
    dist.set_sigma(new_sigma);
    auto sigma = dist.sigma();
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_cols),dist.num_dim());
    EXPECT_TRUE(arma::all(arma::all(arma::symmatu(sigma) == arma::symmatu(new_sigma))));
}

TYPED_TEST(MultivariateNormalDistTest, sigma_inv) {
    auto &dist = this->dist;
    auto sigma = dist.sigma();
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_cols),dist.num_dim());
    auto sigma_inv_true = arma::inv_sympd(arma::symmatu(sigma)).eval();
    auto sigma_inv = dist.sigma_inv();
    EXPECT_EQ(static_cast<const arma::uword>(sigma_inv.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma_inv.n_cols),dist.num_dim());
    EXPECT_TRUE(arma::all(arma::all(arma::symmatu(sigma_inv) == arma::symmatu(sigma_inv_true))));
}

TYPED_TEST(MultivariateNormalDistTest, set_sigma_inv) {
    auto &dist = this->dist;
    auto new_sigma = dist.sigma();
    for(IdxT j=0; j<new_sigma.n_cols; j++) for(IdxT i=0; i<=j; i++) new_sigma(i,j) += 1./double(i+3.141*j+1);
    dist.set_sigma(new_sigma);
    auto sigma = dist.sigma();
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_cols),dist.num_dim());
    EXPECT_TRUE(arma::all(arma::all(arma::symmatu(sigma) == arma::symmatu(new_sigma))));
    auto sigma_inv = dist.sigma_inv();
    auto new_sigma_inv = arma::inv_sympd(arma::symmatu(new_sigma)).eval();
    EXPECT_EQ(static_cast<const arma::uword>(sigma_inv.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma_inv.n_cols),dist.num_dim());
    EXPECT_TRUE(arma::all(arma::all(arma::symmatu(sigma_inv) == arma::symmatu(new_sigma_inv))));
}

TYPED_TEST(MultivariateNormalDistTest, set_params) {
    auto &dist = this->dist;
    auto new_mu = dist.mu();
    for(IdxT k=0; k<new_mu.n_elem; k++) new_mu[k] += 1./double(k+1);
    auto new_sigma = dist.sigma();
    for(IdxT j=0; j<new_sigma.n_cols; j++) for(IdxT i=0; i<=j; i++) new_sigma(i,j) += 1./double(i+3.141*j+1);
    dist.set_params(new_mu,new_sigma);
    auto mu = dist.mu();
    auto sigma = dist.sigma();

    EXPECT_EQ(static_cast<const arma::uword>(mu.n_elem),dist.num_dim());
    EXPECT_TRUE(arma::all(mu==new_mu));
    
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_rows),dist.num_dim());
    EXPECT_EQ(static_cast<const arma::uword>(sigma.n_cols),dist.num_dim());
    EXPECT_TRUE(arma::all(arma::all(arma::symmatu(sigma) == arma::symmatu(new_sigma))));
}
