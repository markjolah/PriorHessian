/** @file rng_environment.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 * @brief An environment for use in googletest that enables repeatable randomized testing
 */
#ifndef TEST_HELPERS_RNG_ENVIRONMENT_H
#define TEST_HELPERS_RNG_ENVIRONMENT_H

#include<random>
#include<iostream>
#include<armadillo>
#include "gtest/gtest.h"

namespace test_helper {

class RngEnvironment : public ::testing::Environment 
{
    using SeedT = uint64_t;
    using RngT = std::mt19937_64;
    static const SeedT MAX_SEED = 9999; //Limit seed size to make typing it in on command line as gtest exe argument easier
    SeedT seed = 0;
    RngT rng;
public:
    using IdxT = arma::uword;
    using VecT = arma::Col<double>;
    using MatT = arma::Mat<double>;
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
    
    void SetUp()
    {
        ::testing::Test::RecordProperty("rng_seed",seed);
        std::cout<<">>>>>>>>>>>> To Repeat Use SEED: "<<seed<<"\n";
    }

    //Use saved seed to reset the RNG.  Typically called before each test to make them independent of ordering.
    void reset_rng()
    {
        rng.seed(seed);
    }
    
    double sample_real(double a, double b) 
    {
        if( a == -INFINITY) a=std::numeric_limits<double>::lowest()/10;
        if( b == INFINITY) b=std::numeric_limits<double>::max()/10;
        std::uniform_real_distribution<double> d(a,b);
        return d(rng);
    }

    template<class IntT>
    IntT sample_integer(IntT a, IntT b) 
    {
        std::uniform_int_distribution<IntT> d(a,b);
        return d(rng);
    }

    double sample_normal(double mean=0, double sigma=1) 
    {
        std::normal_distribution<double> d(mean,sigma);
        return d(rng);
    }

    VecT sample_normal_vec(IdxT N, double mean=0, double sigma=1) 
    {
        VecT v(N);
        std::normal_distribution<double> d(mean,sigma);
        for(IdxT n=0; n<N; n++) v(n) = d(rng);
        return v;
    }
    
    double sample_exponential(double lambda=1) 
    {
        std::exponential_distribution<double> d(lambda);
        return d(rng);
    }

    VecT sample_exponential_vec(IdxT N, double lambda=1) 
    {
        VecT v(N);
        std::exponential_distribution<double> d(lambda);
        for(IdxT n=0; n<N; n++) v(n) = d(rng);
        return v;
    }

    double sample_gamma(double shape, double scale) 
    {
        std::gamma_distribution<double> d(shape,scale);
        return d(rng);
    }

    VecT sample_gamma_vec(IdxT N, double shape=1, double scale=1) 
    {
        VecT v(N);
        std::gamma_distribution<double> d(shape,scale);
        for(IdxT n=0; n<N; n++) v(n) = d(rng);
        return v;
    }
    
    MatT sample_orthonormal_mat(IdxT N)
    {
        MatT R(N,N);
        std::normal_distribution<double> d(0,1);
        for(IdxT i=0; i<N; i++) for(IdxT j=0; j<N; j++) R(j,i) = d(rng);
        return arma::orth(R);
    }
    
    MatT sample_sigma_mat(const VecT &sigma)
    {
        auto P = sample_orthonormal_mat(sigma.n_elem);
        return P.t()*arma::diagmat(sigma)*P;
    }

    RngT& get_rng()
    {
        return rng;
    }
};

} /* namespace test_helper */

#endif /* TEST_HELPERS_RNG_ENVIRONMENT_H */
