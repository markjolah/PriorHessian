/** @file test_CompositeDists.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include <cmath>
#include "gtest/gtest.h"

#include "test_prior_hessian.h"
#include "test_multivariate.h"
#include "test_univariate.h"
#include "PriorHessian/BoundsAdaptedDist.h"
#include "PriorHessian/CompositeDist.h"

using namespace prior_hessian;

namespace detail {
    template<class... Ts, size_t... Is>
    CompositeDist construct_from_tuple(const std::tuple<Ts...> &t, std::index_sequence<Is...>)
    { return CompositeDist{ std::get<Is>(t)... }; }

    template<class... Ts, size_t... Is>
    CompositeDist construct_from_tuple(std::tuple<Ts...> &&t, std::index_sequence<Is...>)
    { return CompositeDist{ std::get<Is>(std::move(t))... }; }
    
    template<class... Ts, size_t... Is>
    void initialize_from_dists(CompositeDist &dist, const std::tuple<Ts...>&ts, std::index_sequence<Is...>)
    { dist.initialize(std::get<Is>(ts)...); }

    template<class... Ts, size_t... Is>
    void initialize_from_dists(CompositeDist &dist, std::tuple<Ts...>&&ts, std::index_sequence<Is...>)
    { dist.initialize(std::get<Is>(ts)...); }
}

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



template<class Dist>
class UnivariateCompositeComponentTest : public ::testing::Test {
public:    
    Dist dist;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};

TYPED_TEST_CASE(UnivariateCompositeComponentTest, UnivariateDistTs);

TYPED_TEST(UnivariateCompositeComponentTest, make_component_dist)
{
    SCOPED_TRACE("make_component_dist");
    auto &dist = this->dist;
    auto comp_dist= CompositeDist::make_component_dist(dist);
    check_equal(dist,comp_dist);
}

template<class Dist>
class MultivariateCompositeComponentTest : public ::testing::Test {
public:    
    Dist dist;
    virtual void SetUp() override {
        env->reset_rng();
        dist = make_dist<Dist>();
    }
};

TYPED_TEST_CASE(MultivariateCompositeComponentTest, MultivariateDistTs);

TYPED_TEST(MultivariateCompositeComponentTest, make_adapted_bounded_dist)
{
    SCOPED_TRACE("make_adapted_bounded_dist");
    auto &dist = this->dist;
    auto b_dist= make_adapted_bounded_dist(dist);
    check_equal(dist,b_dist);
}

TYPED_TEST(MultivariateCompositeComponentTest, make_adapted_bounded_dist_copy)
{
    SCOPED_TRACE("make_adapted_bounded_dist_copy");
    auto &dist = this->dist;
    BoundsAdaptedDistT<TypeParam> b_dist{make_adapted_bounded_dist(dist)};
    check_equal(dist,b_dist);
}


TYPED_TEST(MultivariateCompositeComponentTest, make_component_adapted_bounded_dist)
{
    SCOPED_TRACE("make_component_adapted_bounded_dist");
    auto &dist = this->dist;
    CompositeDist::ComponentDistT<TypeParam> c_dist{make_adapted_bounded_dist(dist)};
    check_equal(dist,c_dist);
}

TYPED_TEST(MultivariateCompositeComponentTest, make_component_dist)
{
    SCOPED_TRACE("make_component_adapted_bounded_dist");
    auto &dist = this->dist;
    auto c_dist=CompositeDist::make_component_dist(dist);
    check_equal(dist,c_dist);
}

void check_composite_dists_equal(CompositeDist &d1, CompositeDist &d2)
{
    ASSERT_EQ((bool)d1,(bool)d2);
    auto Ndim = d1.num_dim();
    ASSERT_EQ(Ndim, d2.num_dim());
    ASSERT_EQ(d1.num_components(), d2.num_components());
    if(d1) { //Test rng-repeatability
        env->reset_rng();
        auto v = d1.sample(env->get_rng());
        env->reset_rng();
        auto v2 = d2.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++) EXPECT_EQ(v[i],v2[i]);
    }    
    ASSERT_EQ(d1,d2);
}


template<class TupleT>
class CompositeDistTest : public ::testing::Test {
public:    
    TupleT dists;
    CompositeDist composite;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        initialize_distribution_tuple(dists);
        composite.initialize(dists);
    }
};


/* 
 * List of type tuples to store in a composite dist 
 * Tests the empty tuple base-case also
 */
using CompositeDistTestTs = ::testing::Types<
    std::tuple<NormalDist, GammaDist, ParetoDist, SymmetricBetaDist>,
    std::tuple<TruncatedDist<NormalDist>, NormalDist, TruncatedDist<GammaDist>>,
    std::tuple<UpperTruncatedDist<ParetoDist>, TruncatedParetoDist, GammaDist, ScaledSymmetricBetaDist>,
    std::tuple<ParetoDist>,
    std::tuple<>,
    std::tuple<MultivariateNormalDist<2>>,
    std::tuple<TruncatedMultivariateNormalDist<2>>,
    std::tuple<MultivariateNormalDist<4>>,
    std::tuple<NormalDist,MultivariateNormalDist<2>>,
    std::tuple<NormalDist,MultivariateNormalDist<2>,TruncatedGammaDist,TruncatedMultivariateNormalDist<2>>
    >;
                                          
TYPED_TEST_CASE(CompositeDistTest, CompositeDistTestTs);

TYPED_TEST(CompositeDistTest, copy_construction) {
    CompositeDist &composite = this->composite;
    IdxT Nparams = composite.num_params();
    IdxT Ndim = composite.num_dim();
    auto params = composite.params();
    auto lbound = composite.lbound();
    CompositeDist dist_copy{composite};  //copy construct
    EXPECT_EQ(dist_copy.num_dim(), Ndim);
    EXPECT_EQ(dist_copy.num_params(), Nparams);
    //Check copy of parameters is successful
    auto params_copy = dist_copy.params();
    ASSERT_EQ(params_copy.n_elem, Nparams);
    for(IdxT i=0; i < Nparams; i++)
        EXPECT_EQ(params[i], params_copy[i]);
    auto lbound_copy = dist_copy.lbound();
    ASSERT_EQ(lbound_copy.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(lbound[i], lbound_copy[i])<<i;
    //Check repeatability of rng generation
    if(composite){
        env->reset_rng();
        auto v = composite.sample(env->get_rng());
        env->reset_rng();
        auto v2 = dist_copy.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++)
            EXPECT_EQ(v[i],v2[i]);
    }
}

TYPED_TEST(CompositeDistTest, copy_assignment) {
    CompositeDist &composite = this->composite;
    CompositeDist dist_copy{};
    volatile double _foo=0;
    _foo = (dist_copy.num_dim(),_foo); //Force something to happen with dist_copy first
    
    dist_copy = composite;  //Move construct
    EXPECT_EQ(dist_copy.num_dim(), composite.num_dim());
    EXPECT_EQ(dist_copy.num_params(), composite.num_params());
    //Check copy of parameters is successful
    auto params = composite.params();
    auto lbound = composite.lbound();
    IdxT Nparams = composite.num_params();
    IdxT Ndim = composite.num_dim();
    auto params_copy = dist_copy.params();
    ASSERT_EQ(params_copy.n_elem, Nparams);
    for(IdxT i=0; i < Nparams; i++)
        EXPECT_EQ(params[i], params_copy[i]);
    auto lbound_copy = dist_copy.lbound();
    ASSERT_EQ(lbound_copy.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(lbound[i], lbound_copy[i])<<i;
    //Check repeatability of rng generation
    if(composite) {
        env->reset_rng();
        auto v = composite.sample(env->get_rng());
        env->reset_rng();
        auto v2 = dist_copy.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++) EXPECT_EQ(v[i],v2[i]);
    }
}


TYPED_TEST(CompositeDistTest, move_construction) {
    CompositeDist &composite = this->composite;
    auto params = composite.params();
    auto lbound = composite.lbound();
    env->reset_rng();
    VecT v;
    if(composite) v = composite.sample(env->get_rng());
    IdxT Nparams = composite.num_params();
    IdxT Ndim = composite.num_dim();
    CompositeDist dist_copy{std::move(composite)};  //Move construct
    EXPECT_EQ(dist_copy.num_dim(), Ndim);
    EXPECT_EQ(dist_copy.num_params(), Nparams);
    //Check copy of parameters is successful
    auto params_copy = dist_copy.params();
    ASSERT_EQ(params_copy.n_elem, Nparams);
    for(IdxT i=0; i < Nparams; i++)
        EXPECT_EQ(params[i], params_copy[i]);
    auto lbound_copy = dist_copy.lbound();
    ASSERT_EQ(lbound_copy.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(lbound[i], lbound_copy[i])<<i;
    //Check repeatability of rng generation
    if(Ndim>0) {
        env->reset_rng();
        auto v2 = dist_copy.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++)
            EXPECT_EQ(v[i],v2[i]);
    }
}

TYPED_TEST(CompositeDistTest, move_assignment) {
    CompositeDist &composite = this->composite;
    CompositeDist dist_copy{std::make_tuple(prior_hessian::NormalDist{})}; //Make something useful to force compiler to do something.
    auto params = composite.params();
    auto lbound = composite.lbound();
    env->reset_rng();
    VecT v;
    if(composite) v = composite.sample(env->get_rng());
    IdxT Nparams = composite.num_params();
    IdxT Ndim = composite.num_dim();
    dist_copy = std::move(composite); //Now move over it with our test fixture dist
    //check basic constants are preserved
    EXPECT_EQ(dist_copy.num_dim(), Ndim);
    EXPECT_EQ(dist_copy.num_params(), Nparams);
    //Check copy of parameters and lbound is successful
    auto params_copy = dist_copy.params();
    ASSERT_EQ(params_copy.n_elem, Nparams);
    for(IdxT i=0; i < Nparams; i++)
        EXPECT_EQ(params[i], params_copy[i]);
    auto lbound_copy = dist_copy.lbound();    
    ASSERT_EQ(lbound_copy.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(lbound[i], lbound_copy[i])<<i;
    //Check repeatability of rng generation
    if(Ndim>0) {
        env->reset_rng();
        auto v2 = dist_copy.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++)
            EXPECT_EQ(v[i],v2[i]);
    }
}

//Check lazily constructed name lists are preserved on copy_construction
TYPED_TEST(CompositeDistTest, names_copy_construction) {
    CompositeDist &composite = this->composite;
    auto comps = composite.component_names();
    auto vars = composite.dim_variables();
    auto params = composite.param_names();
    for(auto &n: comps) n+="_comp";
    for(auto &n: vars) n+="_var";
    for(auto &n: params) n+="_param";
    composite.set_component_names(comps);
    composite.set_dim_variables(vars);
    composite.set_param_names(params);

    CompositeDist dist_copy{composite};

    auto comps2 = composite.component_names();
    auto vars2 = composite.dim_variables();
    auto params2 = composite.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);

    comps2 = dist_copy.component_names();
    vars2 = dist_copy.dim_variables();
    params2 = dist_copy.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);
}

//Check lazily constructed name lists are preserved on move_construction
TYPED_TEST(CompositeDistTest, names_move_construction) {
    CompositeDist &composite = this->composite;
    auto comps = composite.component_names();
    auto vars = composite.dim_variables();
    auto params = composite.param_names();
    for(auto &n: comps) n+="_comp";
    for(auto &n: vars) n+="_var";
    for(auto &n: params) n+="_param";
    composite.set_component_names(comps);
    composite.set_dim_variables(vars);
    composite.set_param_names(params);


    auto comps2 = composite.component_names();
    auto vars2 = composite.dim_variables();
    auto params2 = composite.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);

    CompositeDist dist_copy{std::move(composite)};

    comps2 = dist_copy.component_names();
    vars2 = dist_copy.dim_variables();
    params2 = dist_copy.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);
}

//Check lazily constructed name lists are preserved on copy assignment
TYPED_TEST(CompositeDistTest, names_copy_assignment) {
    CompositeDist &composite = this->composite;
    CompositeDist dist_copy{std::make_tuple(prior_hessian::NormalDist{})}; //Make something useful to force compiler to do something.
    auto comps = composite.component_names();
    auto vars = composite.dim_variables();
    auto params = composite.param_names();
    for(auto &n: comps) n+="_comp";
    for(auto &n: vars) n+="_var";
    for(auto &n: params) n+="_param";
    composite.set_component_names(comps);
    composite.set_dim_variables(vars);
    composite.set_param_names(params);

    dist_copy = composite;

    auto comps2 = composite.component_names();
    auto vars2 = composite.dim_variables();
    auto params2 = composite.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);

    comps2 = dist_copy.component_names();
    vars2 = dist_copy.dim_variables();
    params2 = dist_copy.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);
}


//Check lazily constructed name lists are preserved on move assignment
TYPED_TEST(CompositeDistTest, names_move_assignment) {
    CompositeDist &composite = this->composite;
    CompositeDist dist_copy{std::make_tuple(prior_hessian::NormalDist{})}; //Make something useful to force compiler to do something.
    auto comps = composite.component_names();
    auto vars = composite.dim_variables();
    auto params = composite.param_names();
    for(auto &n: comps) n+="_comp";
    for(auto &n: vars) n+="_var";
    for(auto &n: params) n+="_param";
    composite.set_component_names(comps);
    composite.set_dim_variables(vars);
    composite.set_param_names(params);

    auto comps2 = composite.component_names();
    auto vars2 = composite.dim_variables();
    auto params2 = composite.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);

    dist_copy = std::move(composite);

    comps2 = dist_copy.component_names();
    vars2 = dist_copy.dim_variables();
    params2 = dist_copy.param_names();
    ASSERT_EQ(comps.size(),comps2.size());
    for(IdxT i=0; i<comps.size(); i++) EXPECT_EQ(comps[i],comps2[i]);
    ASSERT_EQ(vars.size(),vars2.size());
    for(IdxT i=0; i<vars.size(); i++) EXPECT_EQ(vars[i],vars2[i]);
    ASSERT_EQ(params.size(),params2.size());
    for(IdxT i=0; i<params.size(); i++) EXPECT_EQ(params[i],params2[i]);
}

TYPED_TEST(CompositeDistTest, num_components) {
    CompositeDist &composite = this->composite;
    EXPECT_EQ(composite.num_components(),std::tuple_size<TypeParam>::value);
}


/**
 * Check  constuction from varadic list of dists is same as initialize 
 * ( which is used internally by CompositeDistTest::SetUp(). )
 * 
 */
TYPED_TEST(CompositeDistTest, construct_from_tuple) {
    CompositeDist &composite = this->composite;
    auto new_composite = construct_from_tuple(this->dists);
    ASSERT_EQ((bool)composite,(bool)new_composite);
    auto Ndim = composite.num_dim();
    ASSERT_EQ(Ndim, new_composite.num_dim());
    ASSERT_EQ(composite.num_components(), new_composite.num_components());
    if(composite) { //Test rng-repeatability
        env->reset_rng();
        auto v = composite.sample(env->get_rng());
        env->reset_rng();
        auto v2 = new_composite.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++) EXPECT_EQ(v[i],v2[i]);
    }
}

TYPED_TEST(CompositeDistTest, construct_from_rvalue_tuple) {
    CompositeDist &composite = this->composite;
    auto tup = this->dists;
    auto new_composite = construct_from_tuple(std::move(tup));
    ASSERT_EQ((bool)composite,(bool)new_composite);
    auto Ndim = composite.num_dim();
    ASSERT_EQ(Ndim, new_composite.num_dim());
    ASSERT_EQ(composite.num_components(), new_composite.num_components());
    if(composite) { //Test rng-repeatability
        env->reset_rng();
        auto v = composite.sample(env->get_rng());
        env->reset_rng();
        auto v2 = new_composite.sample(env->get_rng());
        ASSERT_EQ(v2.n_elem, Ndim);
        for(IdxT i=0; i < Ndim; i++) EXPECT_EQ(v[i],v2[i]);
    }
}

TYPED_TEST(CompositeDistTest, initialize_empty) {
    CompositeDist &composite = this->composite;
    composite.initialize();
    ASSERT_FALSE((bool) composite);
    ASSERT_TRUE(composite.is_empty());
    ASSERT_EQ(composite.num_components(),0);
    ASSERT_EQ(composite.num_dim(),0);
}

TYPED_TEST(CompositeDistTest, initialize_empty_tuple) {
    CompositeDist &composite = this->composite;
    std::tuple<> tup;
    composite.initialize(tup);
    ASSERT_FALSE((bool) composite);
    ASSERT_TRUE(composite.is_empty());
    ASSERT_EQ(composite.num_components(),0);
    ASSERT_EQ(composite.num_dim(),0);
}

TYPED_TEST(CompositeDistTest, initialize_empty_rvalue_tuple) {
    CompositeDist &composite = this->composite;
    composite.initialize(std::tuple<>{});
    ASSERT_FALSE((bool) composite);
    ASSERT_TRUE(composite.is_empty());
    ASSERT_EQ(composite.num_components(),0);
    ASSERT_EQ(composite.num_dim(),0);
}


TYPED_TEST(CompositeDistTest, initialize_from_tuple) {
    SCOPED_TRACE("initialize_from_tuple");
    CompositeDist &composite = this->composite;
    CompositeDist new_composite;
    new_composite.initialize(this->dists);
    check_composite_dists_equal(composite,new_composite);
}

TYPED_TEST(CompositeDistTest, initialize_from_rvalue_tuple) {
    SCOPED_TRACE("initialize_from_rvalue_tuple");
    CompositeDist &composite = this->composite;
    CompositeDist new_composite;
    auto tup = this->dists;
    new_composite.initialize(std::move(tup));
    check_composite_dists_equal(composite,new_composite);
}

TYPED_TEST(CompositeDistTest, initialize_from_dists) {
    SCOPED_TRACE("initialize_from_dists");
    CompositeDist &composite = this->composite;
    CompositeDist new_composite;
    initialize_from_dists(new_composite, this->dists);
    check_composite_dists_equal(composite,new_composite);
}


TYPED_TEST(CompositeDistTest, initialize_from_rvalue_dists) {
    SCOPED_TRACE("initialize_from_rvalue_dists");
    CompositeDist &composite = this->composite;
    CompositeDist new_composite;
    auto tup = this->dists;
    initialize_from_dists(new_composite, std::move(tup));
    check_composite_dists_equal(composite,new_composite);
}

TYPED_TEST(CompositeDistTest, clear) {
    auto &composite = this->composite;
    auto copy = composite;
    EXPECT_EQ(composite,copy);
    composite.clear();
    ASSERT_FALSE((bool) composite);
    ASSERT_TRUE(composite.is_empty());
    ASSERT_EQ(composite.num_components(),0);
    ASSERT_EQ(composite.num_dim(),0);
    copy.clear();
    EXPECT_EQ(composite,copy);
}

TYPED_TEST(CompositeDistTest, is_empty_operator_bool) {
    auto &composite = this->composite;
    EXPECT_EQ(!composite.is_empty(), (bool) composite);
    if(composite) {
        EXPECT_LT(0,composite.num_components());
        EXPECT_LT(0,composite.num_dim());
    } else {
        EXPECT_EQ(0,composite.num_components());
        EXPECT_EQ(0,composite.num_dim());
    }
}

template<class... Ts, std::size_t... I> 
IdxT tuple_num_dim(const std::tuple<Ts...> &tup, std::index_sequence<I...>)
{ return meta::sum_in_order<IdxT>( {std::get<I>(tup).num_dim()...}); }

template<class... Ts> 
IdxT tuple_num_dim(const std::tuple<Ts...> &tup)
{ return tuple_num_dim(tup,std::index_sequence_for<Ts...>{}); }

TYPED_TEST(CompositeDistTest, num_dim) {
    CompositeDist &composite = this->composite;
    EXPECT_EQ(composite.num_dim(),tuple_num_dim(this->dists));
}

TYPED_TEST(CompositeDistTest, num_dim_components) {
    CompositeDist &composite = this->composite;
    auto ndim = composite.num_dim_components();
    ASSERT_EQ(composite.num_components(), ndim.size());
    for(IdxT n=0; n< composite.num_components(); n++)
        EXPECT_LT(0,ndim[n]);
}

TYPED_TEST(CompositeDistTest, dim_variables) {
    //Variable names for each dimension
    CompositeDist &composite = this->composite;
    auto vars = composite.dim_variables();
    ASSERT_EQ(composite.num_dim(), vars.size());
    std::unordered_set<std::string> vars_set(vars.begin(), vars.end());
    EXPECT_EQ(vars.size(), vars_set.size())<<"Var names should be unique.";
    for(auto &v:vars) EXPECT_LT(0,v.size())<<"Var should not be empty.";
}

TYPED_TEST(CompositeDistTest, set_dim_variables) {
    CompositeDist &composite = this->composite;
    auto vars = composite.dim_variables();
    for(auto& v:vars) v.append("foo");
    composite.set_dim_variables(vars);
    auto vars2 = composite.dim_variables();
    ASSERT_EQ(composite.num_dim(), vars2.size());
    for(IdxT n=0; n<composite.num_dim(); n++) EXPECT_EQ(vars2[n],vars[n]);
}

/* Bounds */
TYPED_TEST(CompositeDistTest, lbound_ubound) {
    CompositeDist &composite = this->composite;
    auto lb = composite.lbound();
    ASSERT_TRUE(!lb.has_nan());
    auto ub = composite.ubound();
    ASSERT_TRUE(!ub.has_nan());
    ASSERT_TRUE(arma::all(lb<ub))<<"Bad bounds lb:"<<lb.t()<<" ub:"<<ub.t();
}

TYPED_TEST(CompositeDistTest, in_bounds) {
    CompositeDist &composite = this->composite;
    if(!composite) return;
    auto lb = composite.lbound();
    auto ub = composite.ubound();
    VecT old_s;
    for(IdxT n=0;n<this->Ntest;n++) {
        auto s = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(s));
        if(n>0) {ASSERT_TRUE(!arma::all(s==old_s));}
        old_s = s;
    }
}    

TYPED_TEST(CompositeDistTest, set_bounds) {
    CompositeDist &composite = this->composite;
    if(!composite) return;
    auto val1 = composite.sample(env->get_rng());
    auto val2 = composite.sample(env->get_rng());
    auto new_lb = arma::min(val1,val2);
    auto new_ub = arma::max(val1,val2);
    auto lb = composite.lbound();
    auto ub = composite.ubound();
    ASSERT_TRUE(arma::all(new_lb>lb));
    ASSERT_TRUE(arma::all(new_ub<ub));
    ASSERT_TRUE(composite.in_bounds(new_lb));
    ASSERT_TRUE(composite.in_bounds(new_ub));

    composite.set_lbound(new_lb);
    ASSERT_TRUE(arma::all(new_lb==composite.lbound()));
    composite.set_lbound(lb);
    ASSERT_TRUE(arma::all(lb==composite.lbound()));
    ASSERT_TRUE(composite.in_bounds(new_lb));
    ASSERT_TRUE(!composite.in_bounds(lb) || arma::all(lb==composite.lbound()));

    composite.set_ubound(new_ub);
    ASSERT_TRUE(arma::all(new_ub==composite.ubound()));
    composite.set_ubound(ub);
    ASSERT_TRUE(arma::all(ub==composite.ubound()));
    ASSERT_TRUE(composite.in_bounds(new_ub));
    ASSERT_TRUE(!composite.in_bounds(ub) || arma::all(ub==composite.ubound()));

    composite.set_bounds(new_lb, new_ub);
    ASSERT_TRUE(arma::all(new_lb==composite.lbound()));
    ASSERT_TRUE(arma::all(new_ub==composite.ubound()));
    composite.set_bounds(lb,ub);
    ASSERT_TRUE(arma::all(lb==composite.lbound()));
    ASSERT_TRUE(arma::all(ub==composite.ubound()));
    ASSERT_TRUE(composite.in_bounds(new_lb));
    ASSERT_TRUE(!composite.in_bounds(lb) || arma::all(lb==composite.lbound()));
    ASSERT_TRUE(composite.in_bounds(new_ub));
    ASSERT_TRUE(!composite.in_bounds(ub) || arma::all(ub==composite.ubound()));
}
/*
TYPED_TEST(CompositeDistTest, in_bounds_set_bounds) {
    CompositeDist &composite = this->composite;
    if(!composite) return;
    auto lb = composite.lbound();
    auto ub = composite.ubound();
    auto val1 = composite.sample(env->get_rng());
    auto val2 = composite.sample(env->get_rng());
    double c1 = composite.cdf(val1);
    double c2 = composite.cdf(val2);
    
    auto new_lb = arma::min(val1,val2);
    auto new_ub = arma::max(val1,val2);
    double d1 = composite.cdf(new_lb);
    double d2 = composite.cdf(new_ub);
    ASSERT_LE(d1,c1);
    ASSERT_LE(d1,c2);
    ASSERT_LE(c1,d2);
    ASSERT_LE(c2,d2);

//     std::cout<<"val1: "<<val1.t()<<" c1:"<<c1<<"\n";
//     std::cout<<"val2: "<<val2.t()<<" c2:"<<c2<<"\n";
//     std::cout<<"new_lb: "<<new_lb.t()<<" d1:"<<d1<<"\n";
//     std::cout<<"new_ub: "<<new_ub.t()<<" d2:"<<d2<<"\n";

    
    composite.set_bounds(new_lb, new_ub);
    VecT old_s;
    for(IdxT n=0;n<this->Ntest;n++) {
        auto s = composite.sample(env->get_rng());
//         std::cout<<"s:"<<s.t();
        ASSERT_TRUE(composite.in_bounds(s));
        if(n>0) {ASSERT_TRUE(!arma::all(s==old_s));}
        old_s = s;
    }
}    */

/* Distribution Parameters */
TYPED_TEST(CompositeDistTest, num_params_components) {
    CompositeDist &composite = this->composite;
    auto n_p = composite.num_params_components();
    ASSERT_EQ(composite.num_components(), n_p.size());
    for(auto n: n_p) EXPECT_LT(0,n)<<"Component distributions should have 1 or more parameters";
    EXPECT_EQ(composite.num_params(), arma::sum(n_p))<<"Composite number of parameters should match sum of components.";
}

TYPED_TEST(CompositeDistTest, params_equal_params_components) {
    CompositeDist &composite = this->composite;
    auto cps = composite.params_components();
    auto params = composite.params();
    auto ncps = composite.num_params_components();
    ASSERT_EQ(composite.num_components(), cps.size());
    ASSERT_EQ(composite.num_params(), params.n_elem);
    IdxT pidx=0;
    for(IdxT n=0;n<composite.num_components(); n++) {
        ASSERT_EQ(cps[n].n_elem, ncps[n]);
        for(IdxT k=0; k<ncps[n];k++) EXPECT_EQ(cps[n][k], params[pidx++]);
    }
}

TYPED_TEST(CompositeDistTest, check_params) {
    CompositeDist &composite = this->composite;
    auto params = composite.params();
    ASSERT_TRUE(composite.check_params(params));
}

TYPED_TEST(CompositeDistTest, set_params_idempotent) {
    CompositeDist &composite = this->composite;
    auto params = composite.params();
    ASSERT_TRUE(composite.check_params(params));
    composite.set_params(params);
    auto params2 = composite.params();
    ASSERT_TRUE(composite.check_params(params2));
    ASSERT_TRUE(arma::all(params==params2));
}

TYPED_TEST(CompositeDistTest, params_lbound_ubound) {
    CompositeDist &composite = this->composite;
    auto params = composite.params();
    auto lb = composite.params_lbound();
    auto ub = composite.params_ubound();
    ASSERT_EQ(composite.num_params(), lb.n_elem);
    ASSERT_EQ(composite.num_params(), ub.n_elem);
    ASSERT_TRUE(arma::all(lb <= params))<<"OOB Params: "<<params<<" lb:"<<lb;
    ASSERT_TRUE(arma::all(ub >= params))<<"OOB Params: "<<params<<" ub:"<<ub;
}

// TYPED_TEST(CompositeDistTest, set_params_random) {
//     SCOPED_TRACE("set_params_random");
//     CompositeDist &composite = this->composite;
//     TypeParam new_dists;
//     initialize_distribution_tuple(new_dists);
//     CompositeDist new_composite;
//     new_composite.initialize(new_dists);
//     auto new_params = new_composite.params();
//     std::cout<<"New_params:"<<new_params.t();
//     ASSERT_TRUE(composite.check_params(new_params));
//     composite.set_params(new_params);
//     ASSERT_TRUE(arma::all(new_composite.params()==composite.params()));
//     EXPECT_EQ(composite,new_composite);
// }

TYPED_TEST(CompositeDistTest, param_names) {
    CompositeDist &composite = this->composite;
    auto names = composite.param_names();
    ASSERT_EQ(composite.num_params(), names.size());
    std::unordered_set<std::string> names_set(names.begin(), names.end());
    EXPECT_EQ(names.size(), names_set.size())<<"Param names should be unique.";
    for(auto &n:names) EXPECT_LT(0,n.size())<<"Param Name should not be empty.";
}

TYPED_TEST(CompositeDistTest, param_value_and_index) {
    CompositeDist &composite = this->composite;
    auto names = composite.param_names();
    auto params = composite.params();
    for(auto &n:names) {
        ASSERT_TRUE(composite.has_param(n));
        auto k = composite.get_param_index(n);
        auto v = composite.get_param_value(n);
        ASSERT_EQ(params(k),v)<<"Param index and value do not match";
    }
}
/*
TYPED_TEST(CompositeDistTest, param_set_by_name) {
    CompositeDist &composite = this->composite;
    std::cout<<composite<<"\n";

    TypeParam new_dists;
    initialize_distribution_tuple(new_dists);
    CompositeDist new_composite(new_dists);
    auto new_params = new_composite.params();
    
    auto names = composite.param_names();
    IdxT k=0;
    for(auto &n:names) composite.set_param_value(n,new_params[k++]);
    std::cout<<composite<<"\n";
    ASSERT_TRUE(arma::all(new_params == composite.params()))<<"Individual param setting did not work.";
}*/

TYPED_TEST(CompositeDistTest, cdf) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto cdf = composite.cdf(v);
        ASSERT_TRUE(std::isfinite(cdf));
        ASSERT_LE(0,cdf);
        ASSERT_LE(cdf,1);
    }
}

TYPED_TEST(CompositeDistTest, pdf) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto pdf = composite.pdf(v);
        ASSERT_TRUE(std::isfinite(pdf));
        ASSERT_LE(0,pdf);
    }
}

TYPED_TEST(CompositeDistTest, llh) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto llh = composite.llh(v);
        ASSERT_TRUE(std::isfinite(llh))<<"sample: "<<v.t()<<" params:"<<composite.params()<<" LLH:"<<llh;
    }
}

TYPED_TEST(CompositeDistTest, rllh) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    double old_delta;
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto rllh = composite.rllh(v);
        ASSERT_TRUE(std::isfinite(rllh));
        auto llh = composite.llh(v);
        double delta = llh - rllh;
        if(n>0) {
            EXPECT_FLOAT_EQ(delta,old_delta)<<"Incosistent delta betweeen rllh:"<<rllh<<" and llh:"<<llh;
        }
        old_delta = delta;
    }
}

TYPED_TEST(CompositeDistTest, grad) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad = composite.grad(v);
        ASSERT_EQ(grad.n_elem,composite.num_dim());
        ASSERT_TRUE(grad.is_finite());
    }
}

TYPED_TEST(CompositeDistTest, grad2) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad2 = composite.grad2(v);
        ASSERT_EQ(grad2.n_elem,composite.num_dim());
        ASSERT_TRUE(grad2.is_finite());
    }
}

TYPED_TEST(CompositeDistTest, hess) {
    SCOPED_TRACE("hess");
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto hess = composite.hess(v);
        ASSERT_TRUE(hess.is_finite());
        ASSERT_EQ(hess.n_rows,composite.num_dim());
        ASSERT_EQ(hess.n_cols,composite.num_dim());
    }
}

TYPED_TEST(CompositeDistTest, grad_accumulate) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad = composite.grad(v);
        ASSERT_EQ(grad.n_elem,composite.num_dim());
        auto grad_acc = composite.make_zero_grad();
        ASSERT_EQ(grad_acc.n_elem,composite.num_dim());
        ASSERT_TRUE(arma::all(grad_acc==0))<<"Grad should be initialized to 0";
        composite.grad_accumulate(v,grad_acc);
        ASSERT_TRUE(grad_acc.is_finite());
        ASSERT_TRUE(arma::all(grad_acc == grad))<<"Grad should matach grad_accumulate";
    }
}

TYPED_TEST(CompositeDistTest, grad2_accumulate) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad2 = composite.grad2(v);
        ASSERT_EQ(grad2.n_elem,composite.num_dim());
        auto grad2_acc = composite.make_zero_grad();
        ASSERT_EQ(grad2_acc.n_elem,composite.num_dim());
        ASSERT_TRUE(arma::all(grad2_acc==0))<<"Grad should be initialized to 0";
        composite.grad2_accumulate(v,grad2_acc);
        ASSERT_TRUE(grad2_acc.is_finite());
        ASSERT_TRUE(arma::all(grad2_acc == grad2))<<"Grad2 should matach grad2_accumulate";
    }
}

TYPED_TEST(CompositeDistTest, hess_accumulate) {
    SCOPED_TRACE("hess_accumulate");
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto hess = composite.hess(v);
        ASSERT_EQ(hess.n_rows,composite.num_dim());
        ASSERT_EQ(hess.n_cols,composite.num_dim());
        auto hess_acc = composite.make_zero_hess();
        ASSERT_EQ(hess_acc.n_rows,composite.num_dim());
        ASSERT_EQ(hess_acc.n_cols,composite.num_dim());
        ASSERT_EQ(0,arma::abs(hess_acc).max())<<"Hess should be initialized to 0";
        composite.hess_accumulate(v,hess_acc);
        ASSERT_TRUE(hess_acc.is_finite());
        ASSERT_TRUE(arma::approx_equal(hess,hess_acc,"reldiff",1e-8))<<"Hess should matach hess_accumulate";
//         check_symmetric(hess_acc);  We return upper-triangular matrix form
    }
}

TYPED_TEST(CompositeDistTest,grad_grad2_accumulate) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad = composite.grad(v);
        auto grad2 = composite.grad2(v);
        auto grad_acc = composite.make_zero_grad();
        auto grad2_acc = composite.make_zero_grad();
        
        composite.grad_grad2_accumulate(v,grad_acc,grad2_acc);
        ASSERT_TRUE(grad_acc.is_finite());
        ASSERT_TRUE(grad2_acc.is_finite());
        ASSERT_TRUE(arma::approx_equal(grad,grad_acc,"reldiff",1e-8))<<"Grad should matach grad_accumulate";        
        ASSERT_TRUE(arma::approx_equal(grad2,grad2_acc,"reldiff",1e-8))<<"Grad2:"<<grad2.t()<<" should matach grad2_accumulate:"<<grad2_acc.t();
    }
}

TYPED_TEST(CompositeDistTest,grad_hess_accumulate) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto grad = composite.grad(v);
        auto hess = composite.hess(v);
        auto grad_acc = composite.make_zero_grad();
        auto hess_acc = composite.make_zero_hess();
        
        composite.grad_hess_accumulate(v,grad_acc,hess_acc);
        ASSERT_TRUE(grad_acc.is_finite());
        ASSERT_TRUE(hess_acc.is_finite());
        ASSERT_TRUE(arma::approx_equal(grad,grad_acc,"reldiff",1e-8))<<"Grad should matach grad_accumulate";        
        ASSERT_TRUE(arma::approx_equal(hess,hess_acc,"reldiff",1e-8))<<"Hess2:"<<hess<<" should matach hess_accumulate:"<<hess_acc;
    }
}

TYPED_TEST(CompositeDistTest, rllh_components) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto rllh = composite.rllh(v);
        auto rllh_components = composite.rllh_components(v);
        ASSERT_TRUE(std::isfinite(rllh));
        ASSERT_EQ(rllh_components.n_elem, composite.num_components());
        ASSERT_FLOAT_EQ(rllh,arma::sum(rllh_components))<<"rllh components should sum to:"<<arma::sum(rllh_components)<<" but should be rllh:"<<rllh;
    }
}

TYPED_TEST(CompositeDistTest, llh_components) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    for(IdxT n=0; n<this->Ntest; n++) {
        auto v = composite.sample(env->get_rng());
        ASSERT_TRUE(composite.in_bounds(v));
        auto llh = composite.llh(v);
        auto llh_components = composite.llh_components(v);
        ASSERT_TRUE(std::isfinite(llh))<<"v:"<<v.t()<<" params:"<<composite.params().t()<<" llh: "<<llh<<" comps:"<<llh_components.t();
        ASSERT_EQ(llh_components.n_elem, composite.num_components());
        ASSERT_FLOAT_EQ(llh,arma::sum(llh_components))<<"llh components should sum to:"<<arma::sum(llh_components)<<" but should be llh:"<<llh;
    }
}

TYPED_TEST(CompositeDistTest, sample_repeatablity) {
    CompositeDist &composite = this->composite;
    if(!composite) return; //Ignore empty dists.
    auto& rng = env->get_rng();
    CompositeDist::AnyRngT any_rng(rng);
    env->reset_rng();
    auto v11 = composite.sample(rng);
    auto v12 = composite.sample(any_rng);
    ASSERT_TRUE(composite.in_bounds(v11));
    ASSERT_TRUE(composite.in_bounds(v12));
    env->reset_rng();
    auto v21 = composite.sample(any_rng);
    auto v22 = composite.sample(rng);
    EXPECT_TRUE(arma::all(v11==v21))<<"Random number generation not repeatable.";
    EXPECT_TRUE(arma::all(v12==v22))<<"Random number generation not repeatable.";
}

TYPED_TEST(CompositeDistTest, bulk_sample_repeatablity) {
    CompositeDist &composite = this->composite;
    auto Ntest = this->Ntest;
    if(!composite) return; //Ignore empty dists.
    auto& rng = env->get_rng();
    CompositeDist::AnyRngT any_rng(rng);
    env->reset_rng();
    auto v11 = composite.sample(rng,Ntest);
    auto v12 = composite.sample(any_rng,Ntest);
    ASSERT_TRUE(composite.in_bounds_all(v11));
    ASSERT_TRUE(composite.in_bounds_all(v12));
    env->reset_rng();
    auto v21 = composite.sample(any_rng,Ntest);
    auto v22 = composite.sample(rng,Ntest);
    EXPECT_TRUE(arma::approx_equal(v11,v21,"absdiff",0))<<"Random number generation not repeatable."<<v11<<" "<<v21;
    EXPECT_TRUE(arma::approx_equal(v12,v22,"absdiff",0))<<"Random number generation not repeatable."<<v12<<" "<<v22;
}
