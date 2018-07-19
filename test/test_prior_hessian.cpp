/** @file test_prior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief Main google test for prior_hessian
 */
#include "test_prior_hessian.h"

/* Globals */
test_helper::RngEnvironment *env = new test_helper::RngEnvironment; //Googletest wants to free env, so we need to appease its demands or face segfaults.


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
       
    ::testing::AddGlobalTestEnvironment(env);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
