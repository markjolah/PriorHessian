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
#include "PriorHessian/SymmetricBetaDist.h"
#include "PriorHessian/CompositeDist.h"
#include <random>

using namespace prior_hessian;

/* Globals */
extern test_helper::RngEnvironment *env;

/* Factory functions */
template<class Dist> 
Dist make_dist();
template<> NormalDist make_dist();
template<> GammaDist make_dist();
template<> ParetoDist make_dist();
template<> SymmetricBetaDist make_dist();

/* Type parameterized test fixtures */
template<class Dist>
class UnivariateDistTest : public ::testing::Test {
public:    
    Dist dist;
    static constexpr int Ntest = 100;
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
    static constexpr double mu0 = -10.0;
    static constexpr double sigma0 = 7.0;
    static constexpr double theta1 = 10.0;
    static constexpr double kappa1 = 3.0;
    
    static constexpr double lbound2 = 1.0;
    static constexpr double alpha2 = 3.0;
    static constexpr double beta3 = 2.0;
    static constexpr int Ntest = 100;
    
    
    std::tuple<NormalDist,GammaDist,ParetoDist,SymmetricBetaDist> dists;
//     std::tuple<BoundedNormalDist,BoundedGammaDist,BoundedParetoDist,ScaledSymmetricBetaDist> adapted_dists;
    
    using CompositeDistT = CompositeDist<RngT>;
    
    CompositeDistT composite;
    
    
    
    CompositeDistCompositionTest() 
        : dists{std::make_tuple(NormalDist{mu0,sigma0}, GammaDist{theta1,kappa1}, ParetoDist{alpha2,lbound2}, SymmetricBetaDist{beta3})},
          composite{dists}
    {           
//         adapted_dists = make_adapted_bounded_dist_tuple(dists);
    }
};
