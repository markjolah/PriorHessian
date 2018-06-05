/** @file test_prior_hessian.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Common include for all testing modules
 */
#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"

#include "PriorHessian/UnivariateDist.h"
#include "PriorHessian/NormalDist.h"
#include "PriorHessian/GammaDist.h"
#include "PriorHessian/ParetoDist.h"
#include "PriorHessian/CompositeDist.h"
#include <random>
/* Globals */
extern test_helper::RngEnvironment *env;

/* Factory functions */
template<class Dist> 
Dist make_dist();
template<> prior_hessian::NormalDist make_dist();

template<class Dist> 
Dist make_dist();
template<> prior_hessian::NormalDist make_dist();

/* Type parameterized test fixtures */
template<class Dist>
class UnivariateDistTest : public ::testing::Test {
public:    
    Dist dist{0,1,"x"};
    virtual void SetUp() {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};



//Test a CompositeDist where only a single dist (dimension) is used
// template<class Dist>
// class CompositeDistSingletonTest : public ::testing::Test {
// public:    
//     Dist dist{0,1,"x"};
//     virtual void SetUp() {
//         env->reset_rng();
//         dist = make_dist<Dist>();
//     }
// };

//Test the iteration technology of Composite Dist matches the individual items  
class CompositeDistCompositionTest : public ::testing::Test {
public:
    using RngT=std::mt19937_64;
    static constexpr double mean0 = -10.0;
    static constexpr double sigma0 = 7.0;
    static constexpr double mean1 = 10.0;
    static constexpr double kappa1 = 3.0;
    
    static constexpr double lbound2 = 1.0;
    static constexpr double alpha2 = 3.0;
    prior_hessian::NormalDist dist0{mean0,sigma0,"x"};
    prior_hessian::GammaDist dist1{mean1,kappa1,"y"};
    prior_hessian::ParetoDist dist2{alpha2,lbound2,"z"};
    prior_hessian::CompositeDist<RngT> cd;
    CompositeDistCompositionTest() : cd(std::make_tuple(dist0,dist1,dist2)) {}
};
