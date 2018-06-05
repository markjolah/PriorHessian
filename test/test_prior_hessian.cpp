/** @file test_pprior_hessian.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the ParallelRngManager class
 */
#include<random>

#include "gtest/gtest.h"

#include "PriorHessian/NormalDist.h"
#include "BacktraceException/BacktraceException.h"

namespace {

class RngEnvironment : public ::testing::Environment 
{
    using SeedT = uint64_t;
    using RngT = std::mt19937_64;
    static const SeedT MAX_SEED = 9999; //Limit seed size to make typing it in on command line as gtest exe argument easier
    SeedT seed = 0;
    RngT rng;
public:
    SeedT get_seed() const {return seed;}
    
    void set_seed(SeedT _seed)
    {
        seed = _seed;
    }
    
    void set_seed()
    {
        //Generate a small human-typeable seed value.  This will give us good enough coverage and make
        //it easy to enter seeds from the command line
        std::random_device rng;
        std::uniform_int_distribution<SeedT> seed_dist(0,MAX_SEED);
        seed = seed_dist(rng);
    }
    
    void SetUp() const
    {
        ::testing::Test::RecordProperty("rng_seed",seed);
        std::cout<<">>>>>>>>>>>> SEED: "<<seed<<"\n";
    }

    //Use saved seed to reset the RNG.  Typically called before each test to make them independent of ordering.
    void reset_rng()
    {
        rng.seed(seed);
    }
    
    double sample_real(double a, double b) 
    {
        std::uniform_real_distribution<double> d(a,b);
        return d(rng);
    }

    template<class IntT>
    IntT sample_integer(IntT a, IntT b) 
    {
        std::uniform_int_distribution<IntT> d(a,b);
        return d(rng);
    }

    double sample_normal(double mean, double sigma) 
    {
        std::normal_distribution<double> d(mean,sigma);
        return d(rng);
    }
    
    double sample_exponential(double lambda) 
    {
        std::exponential_distribution<double> d(lambda);
        return d(rng);
    }
    
    RngT& get_rng()
    {
        return rng;
    }
};

RngEnvironment *env = new RngEnvironment;
using RngT = std::mt19937_64;


template<class Dist> 
Dist make_dist();

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


template<class Dist>
class UnivariateDistTest : public ::testing::Test {
public:    
    Dist dist{0,1,"x"};
    virtual void SetUp() {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};

using UnivariateDistTs = ::testing::Types<prior_hessian::NormalDist> ;
TYPED_TEST_CASE(UnivariateDistTest, UnivariateDistTs);


TYPED_TEST(UnivariateDistTest, get_params) {
    auto params = this->dist.get_params();    
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    for(size_t i=0; i<this->dist.num_params(); i++)
        EXPECT_EQ(params[i],this->dist.get_param(i));
}
/*
TEST_F(NormalDistTest, set_params) {
    auto new_params = this->dist.get_params();
    new_params[0]+=2;
    new_params[1]+=5;
    this->dist.set_params(new_params);
    auto params = this->dist.get_params();
    EXPECT_EQ(params.n_elem,this->dist.num_params());
    EXPECT_EQ(params[0],new_params[0]);
    EXPECT_EQ(params[1],new_params[1]);
}

TEST_F(NormalDistTest, set_lbound) {
    double lbound = this->dist.get_lbound();
    EXPECT_EQ(lbound,-INFINITY);
    //check set_bounds()
    double new_lbound = -10;
    this->dist.set_bounds(new_lbound,this->dist.get_ubound());
    EXPECT_EQ(new_lbound,this->dist.get_lbound());
    //check set_lbound()
    new_lbound = -20;
    this->dist.set_lbound(new_lbound);
    EXPECT_EQ(new_lbound,this->dist.get_lbound());
}

TEST_F(NormalDistTest, set_ubound) {
    double ubound = this->dist.get_ubound();
    EXPECT_EQ(ubound,INFINITY);
    //check set_bounds()
    double new_ubound = 10;
    this->dist.set_bounds(this->dist.get_lbound(),new_ubound);
    EXPECT_EQ(new_ubound,this->dist.get_ubound());
    //check set_ubound()
    new_ubound = 20;
    this->dist.set_ubound(new_ubound);
    EXPECT_EQ(new_ubound,this->dist.get_ubound());
}

TEST_F(NormalDistTest, set_illegal_bounds) {
    //check set_bounds()
    EXPECT_THROW(this->dist.set_bounds(1,1), prior_hessian::PriorHessianError);
    EXPECT_THROW(this->dist.set_bounds(0,-1), prior_hessian::PriorHessianError);
    this->dist.set_lbound(-1);
    EXPECT_THROW(this->dist.set_ubound(-2), prior_hessian::PriorHessianError);
    this->dist.set_ubound(1);
    EXPECT_THROW(this->dist.set_lbound(2), prior_hessian::PriorHessianError);
}

TEST_F(NormalDistTest, set_bounds_cdf) {
    RngT rng(environment.get_seed());
    std::uniform_real_distribution<double> d(-1e6,1e6);
    double lbound = d(rng);
    d = std::uniform_real_distribution<double>(lbound+std::numeric_limits<double>::epsilon(),1e6);
    double ubound = d(rng);
    this->dist.set_ubound(ubound);
    EXPECT_EQ(this->dist.cdf(ubound),1)<<"ubound: "<<ubound;
    this->dist.set_lbound(lbound);
    EXPECT_EQ(this->dist.cdf(lbound),0)<<"lbound: "<<lbound;
}*/

// TEST_F(NormalDistTest, set_bounds_cdf) {
//     double ubound = 20;
//     this->dist.set_ubound(ubound);
//     EXPECT_EQ(this->dist.cdf(ubound),1);
//     double lbound = 0;
//     this->dist.set_lbound(lbound);
//     EXPECT_EQ(this->dist.cdf(lbound),0);
// }


}  // namespace

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
