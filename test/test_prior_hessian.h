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
#include <type_traits>

using namespace prior_hessian;

/* Globals */
extern test_helper::RngEnvironment *env;

/**** Hyper-priors ****/
/*
 * Use RNG in env to sample from the hyper-prior for testing 
 * 
 * 
 */
template<class Dist, class BaseDist> using DerivedFrom = std::enable_if_t<std::is_base_of<std::decay_t<BaseDist>,std::decay_t<Dist>>::value,std::decay_t<Dist>>;

template<class Dist>
void initialize_dist(DerivedFrom<Dist,NormalDist> &d)
{
    //hyper-params
    double mu_mean = 3.0;
    double mu_sigma = 7.0;
    double sigma_scale = 3.0;
    double sigma_shape = 2.0;
    d.set_param(0,env->sample_normal(mu_mean,mu_sigma));
    d.set_param(1,env->sample_gamma(sigma_scale,sigma_shape));
}

template<class Dist>
void initialize_dist(DerivedFrom<Dist,GammaDist> &d)
{
    //hyper-params
    double scale_scale = 10.0;
    double scale_shape = 1.5;
    double shape_scale = 10.0;
    double shape_shape = 1.5;
    d.set_param(0,env->sample_gamma(scale_scale,scale_shape));
    d.set_param(1,env->sample_gamma(shape_scale,shape_shape));
}

template<class Dist>
void initialize_dist(DerivedFrom<Dist,ParetoDist> &d)
{
    //hyper-params
    double alpha_scale = 2.0;
    double alpha_shape = 2.0;
    double lbound_scale = 10.0;
    double lbound_shape = 1.5;
    d.set_param(0,env->sample_gamma(alpha_scale,alpha_shape));
    d.set_lbound(env->sample_gamma(lbound_scale,lbound_shape));
}

template<class Dist>
void initialize_dist(DerivedFrom<Dist,SymmetricBetaDist> &d)
{
    //hyper-params
    double beta_scale = 10.0;
    double beta_shape = 1.5;
    d.set_param(0,env->sample_gamma(beta_scale,beta_shape));
}

/* Factory function */
template<class Dist> 
Dist make_dist()
{
    Dist dist;
    initialize_dist<Dist>(dist);
    return dist;
}

namespace detail {
    template<class... Ts, size_t... Is>
    void initialize_distribution_tuple(std::tuple<Ts...> &t, std::index_sequence<Is...> )
    { meta::call_in_order<int>({(initialize_dist<Ts>(std::get<Is>(t)),0)... }); }
    
    template<class... Ts, size_t... Is>
    CompositeDist construct_from_tuple(const std::tuple<Ts...> &t, std::index_sequence<Is...>)
    { return CompositeDist{ std::get<Is>(t)... }; }

    template<class... Ts, size_t... Is>
    CompositeDist construct_from_tuple(std::tuple<Ts...> &&t, std::index_sequence<Is...>)
    { return CompositeDist{ std::get<Is>(std::move(t))... }; }
    
    template<class... Ts, size_t... Is>
    void initialize_from_dists(CompositeDist &dist, const std::tuple<Ts...>&ts, std::index_sequence<Is...>)
    { return dist.initialize(std::get<Is>(ts)...); }

    template<class... Ts, size_t... Is>
    void initialize_from_dists(CompositeDist &dist, std::tuple<Ts...>&&ts, std::index_sequence<Is...>)
    { return dist.initialize(std::get<Is>(ts)...); }
}

template<class... Ts>
void initialize_distribution_tuple(std::tuple<Ts...> &t)
{ if(sizeof...(Ts)) ::detail::initialize_distribution_tuple(t,std::index_sequence_for<Ts...>{}); }

template<class... Ts>
CompositeDist construct_from_tuple(const std::tuple<Ts...> &t)
{ return ::detail::construct_from_tuple(t,std::index_sequence_for<Ts...>{}); }

template<class... Ts>
CompositeDist construct_from_tuple(std::tuple<Ts...> &&t)
{ return ::detail::construct_from_tuple(std::move(t),std::index_sequence_for<Ts...>{}); }

template<class... Ts>
void initialize_from_dists(CompositeDist &dist, const std::tuple<Ts...>&ts)
{ return ::detail::initialize_from_dists(dist, ts, std::index_sequence_for<Ts...>{}); }

template<class... Ts>
void initialize_from_dists(CompositeDist &dist, std::tuple<Ts...>&&ts)
{ return ::detail::initialize_from_dists(dist, std::move(ts), std::index_sequence_for<Ts...>{}); }


//Test the iteration technology of Composite Dist matches the individual items
/* Type parameterized test fixtures */
template<class Dist>
class UnivariateDistTest : public ::testing::Test {
public:    
    Dist dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};


template<class TupleT>
class CompositeDistTest : public ::testing::Test {
public:    
    TupleT dists;
    CompositeDist composite;
    
    virtual void SetUp() override {
        env->reset_rng();
        initialize_distribution_tuple(dists);
        composite.initialize(dists);
    }
};


