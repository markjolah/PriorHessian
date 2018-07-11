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
    return prior_hessian::NormalDist(mu,sigma);
}

template<>
prior_hessian::GammaDist make_dist()
{
    double alpha_theta = 10.;
    double beta_theta = 1.;
    double alpha_kappa = 1.;
    double beta_kappa = 3.;
    double theta = env->sample_gamma(alpha_theta,beta_theta);
    double kappa = env->sample_gamma(alpha_kappa,beta_kappa);
    return prior_hessian::GammaDist(theta,kappa);
}

template<>
prior_hessian::ParetoDist make_dist()
{
    double alpha_alpha = 1.;
    double beta_alpha = 10.;
    double alpha_lbound= 1.;
    double beta_lbound = 10.;
    double alpha = env->sample_gamma(alpha_alpha,beta_alpha);
    double lbound = env->sample_gamma(alpha_lbound,beta_lbound);
    return prior_hessian::ParetoDist(alpha,lbound);
}

template<>
prior_hessian::SymmetricBetaDist make_dist()
{
    double alpha_beta = 1;
    double beta_beta = 10.;
    double beta = env->sample_gamma(alpha_beta,beta_beta);
    return prior_hessian::SymmetricBetaDist(beta);
}


int main(int argc, char **argv) 
{
    if(argc>2 && !strncmp("--seed",argv[1],6)){
        char* end;
        env->set_seed(strtoull(argv[2],&end,0));
        argc-=2;
        argv+=2;
    } else {
        env->set_seed();
    }
    ::testing::InitGoogleTest(&argc, argv);
    
    backtrace_exception::disable_backtraces();
    
    ::testing::AddGlobalTestEnvironment(env);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
