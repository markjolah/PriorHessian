/** @file test_prior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Main google test for prior_hessian
 */
#include "test_prior_hessian.h"
#include "test_univariate.h"
#include "test_multivariate.h"
#include "PriorHessian/util.h"

using namespace prior_hessian;

/* Globals */
test_helper::RngEnvironment *env = new test_helper::RngEnvironment; //Googletest wants to free env, so we need to appease its demands or face segfaults.

void check_symmetric(const arma::mat &m)
{
    ASSERT_TRUE(m.is_square());
    IdxT N = m.n_rows;
    for(IdxT c=0; c<N; c++) {
        for(IdxT r=0; r<c; r++){
            EXPECT_EQ(m(r,c), m(c,r))<<"Matrix not symmetric.";
        }
    }
}

void check_positive_definite(const arma::mat &m)
{
    auto R = m;
    ASSERT_TRUE(arma::chol(R,m))<<"Cholesky fail. Matrix not positive definite.";
}


int main(int argc, char **argv)
{
    if(argc>2 && !strncmp("--seed",argv[1],6)){
        char* end;
        auto seed = strtoull(argv[2],&end,0);
        env->set_seed(seed);
        //Pass on seed to G-test as command line argument
        const int N=30;
        char buf[N];
        snprintf(buf,N,"--gtest_random_seed=%llu",seed);
        argv[2] = buf;
        argc--; argv++;
    } else {
        env->set_seed();
    }
    ::testing::AddGlobalTestEnvironment(env);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
