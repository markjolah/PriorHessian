/** @file test_prior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Main google test for prior_hessian
 */
#include "test_prior_hessian.h"
#include "BacktraceException/BacktraceException.h"

/* Globals */
test_helper::RngEnvironment *env = new test_helper::RngEnvironment; //Googletest wants to free env, so we need to appease its demands or face segfaults.

template<>
prior_hessian::NormalDist make_dist()
{
    double mu_mean = 0.0;
    double mu_sigma = 1000.0;
    double sigma_lambda = 1;
    double mu = env->sample_normal(mu_mean,mu_sigma);
    double sigma = env->sample_exponential(sigma_lambda);
    return prior_hessian::NormalDist(mu,sigma,"x");
}

int main(int argc, char **argv) 
{
    if(argc>1) {
        char* end;
        env->set_seed(strtoull(argv[0],&end,0));
    } else {
        env->set_seed();
    }
    
    backtrace_exception::disable_backtraces();
    
    ::testing::AddGlobalTestEnvironment(env);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
