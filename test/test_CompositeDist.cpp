/** @file test_CompositeDists.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the CompositeDist 
 */
#include <cmath>
#include "gtest/gtest.h"

#include "test_prior_hessian.h"

TEST_F(CompositeDistCompositionTest, num_component_dists) {
    EXPECT_EQ(this->cd.num_component_dists(),3);
}

TEST_F(CompositeDistCompositionTest, component_types) {
    auto types = this->cd.component_types();
    ASSERT_EQ(this->cd.num_component_dists(), types.size());
    EXPECT_EQ(std::type_index(typeid(dist0)), types[0]);
    EXPECT_EQ(std::type_index(typeid(dist1)), types[1]);
    EXPECT_EQ(std::type_index(typeid(dist2)), types[2]);
}

TEST_F(CompositeDistCompositionTest, num_dim) {
    EXPECT_EQ(this->cd.num_dim(),3);
}

TEST_F(CompositeDistCompositionTest, components_num_dim) {
    auto ndim = this->cd.components_num_dim();
    ASSERT_EQ(this->cd.num_component_dists(), ndim.size());
    EXPECT_EQ(dist0.num_dim(), ndim[0]);
    EXPECT_EQ(dist1.num_dim(), ndim[1]);
    EXPECT_EQ(dist2.num_dim(), ndim[2]);
}

TEST_F(CompositeDistCompositionTest, dim_variables) {
    auto vars = this->cd.dim_variables();
    ASSERT_EQ(this->cd.num_dim(), vars.size());
    EXPECT_EQ(dist0.var_name(), vars[0]);
    EXPECT_EQ(dist1.var_name(), vars[1]);
    EXPECT_EQ(dist2.var_name(), vars[2]);
}

TEST_F(CompositeDistCompositionTest, set_dim_variables) {
    auto vars = this->cd.dim_variables();
    for(auto& v:vars) v.append("foo");
    this->cd.set_dim_variables(vars);
    auto vars2 = this->cd.dim_variables();
    ASSERT_EQ(this->cd.num_dim(), vars2.size());
    EXPECT_EQ(dist0.var_name()+"foo", vars2[0]);
    EXPECT_EQ(dist1.var_name()+"foo", vars2[1]);
    EXPECT_EQ(dist2.var_name()+"foo", vars2[2]);
}

TEST_F(CompositeDistCompositionTest, lbound) {
    auto lbound = this->cd.lbound();
    ASSERT_EQ(this->cd.num_dim(), lbound.size());
    EXPECT_EQ(dist0.lbound(), lbound[0]);
    EXPECT_EQ(dist1.lbound(), lbound[1]);
    EXPECT_EQ(dist2.lbound(), lbound[2]);
}

TEST_F(CompositeDistCompositionTest, set_lbound) {
    auto lbound = this->cd.lbound();
    for(size_t i=0;i<lbound.size();i++) lbound[i]=3.141+i;
    this->cd.set_lbound(lbound);
    auto lbound2 = this->cd.lbound();
    ASSERT_EQ(this->cd.num_dim(), lbound2.size());
    EXPECT_EQ(3.141+0, lbound2[0]);
    EXPECT_EQ(3.141+1, lbound2[1]);
    EXPECT_EQ(3.141+2, lbound2[2]);
}

TEST_F(CompositeDistCompositionTest, ubound) {
    auto ubound = this->cd.ubound();
    ASSERT_EQ(this->cd.num_dim(), ubound.size());
    EXPECT_EQ(dist0.ubound(), ubound[0]);
    EXPECT_EQ(dist1.ubound(), ubound[1]);
    EXPECT_EQ(dist2.ubound(), ubound[2]);
}

TEST_F(CompositeDistCompositionTest, set_ubound) {
    auto ubound = this->cd.ubound();
    for(size_t i=0;i<ubound.size();i++) ubound[i]=3.141+i;
    this->cd.set_ubound(ubound);
    auto ubound2 = this->cd.ubound();
    ASSERT_EQ(this->cd.num_dim(), ubound2.size());
    EXPECT_EQ(3.141+0, ubound2[0]);
    EXPECT_EQ(3.141+1, ubound2[1]);
    EXPECT_EQ(3.141+2, ubound2[2]);
}

TEST_F(CompositeDistCompositionTest, set_bounds) {
    auto lbounds = this->cd.lbound();
    auto ubounds = this->cd.ubound();
    for(size_t i=0;i<lbounds.size();i++) {lbounds[i]=3.141+i; ubounds[i]=i+10;}
    this->cd.set_bounds(lbounds,ubounds);
    auto lbounds2 = this->cd.lbound();
    auto ubounds2 = this->cd.ubound();
    ASSERT_EQ(this->cd.num_dim(), ubounds2.size());
    ASSERT_EQ(this->cd.num_dim(), lbounds2.size());
    EXPECT_EQ(3.141+0, lbounds2[0]);
    EXPECT_EQ(3.141+1, lbounds2[1]);
    EXPECT_EQ(3.141+2, lbounds2[2]);
    EXPECT_EQ(10+0, ubounds2[0]);
    EXPECT_EQ(10+1, ubounds2[1]);
    EXPECT_EQ(10+2, ubounds2[2]);
}

TEST_F(CompositeDistCompositionTest, component_num_params) {
    auto nps = this->cd.components_num_params();
    ASSERT_EQ(this->cd.num_params(), arma::sum(nps));
    EXPECT_EQ(dist0.num_params(), nps[0]);
    EXPECT_EQ(dist1.num_params(), nps[1]);
    EXPECT_EQ(dist2.num_params(), nps[2]);
}

TEST_F(CompositeDistCompositionTest, params) {
    auto params = this->cd.params();
    auto nps = this->cd.components_num_params();
    ASSERT_EQ(this->cd.num_params(), params.size());
    ASSERT_EQ(this->cd.num_component_dists(), nps.size());
    size_t j=0;
    for(size_t k=0; k<nps[0]; k++)
        EXPECT_EQ(dist0.get_param(k),params[j++]);
    for(size_t k=0; k<nps[1]; k++)
        EXPECT_EQ(dist1.get_param(k),params[j++]);
    for(size_t k=0; k<nps[2]; k++)
        EXPECT_EQ(dist2.get_param(k),params[j++]);
}

TEST_F(CompositeDistCompositionTest, set_params) {
    auto params = this->cd.params();
    for(auto& p:params) p*=1.01;
    this->cd.set_params(params);
    auto params2 = this->cd.params();
    ASSERT_EQ(this->cd.num_params(), params2.size());
    for(size_t k=0;k<params2.size();k++) EXPECT_EQ(params[k],params2[k]);
}

TEST_F(CompositeDistCompositionTest, params_desc) {
    auto pd = this->cd.params_desc();
    auto nps = this->cd.components_num_params();
    ASSERT_EQ(this->cd.num_params(), pd.size());
    ASSERT_EQ(this->cd.num_component_dists(), nps.size());
    size_t j=0;
    for(size_t k=0; k<nps[0]; k++)
        EXPECT_EQ(dist0.params_desc()[k],pd[j++]);
    for(size_t k=0; k<nps[1]; k++)
        EXPECT_EQ(dist1.params_desc()[k],pd[j++]);
    for(size_t k=0; k<nps[2]; k++)
        EXPECT_EQ(dist2.params_desc()[k],pd[j++]);
}

TEST_F(CompositeDistCompositionTest, set_params_desc) {
    auto pd = this->cd.params_desc();
    ASSERT_EQ(this->cd.num_params(), pd.size());
    for(auto& p:pd) p.append("abc1234567890");    
    this->cd.set_params_desc(pd);
    auto pd2 = this->cd.params_desc();
    ASSERT_EQ(this->cd.num_params(), pd2.size());
    for(size_t k=0;k<pd2.size();k++) 
        EXPECT_EQ(pd[k],pd2[k]);
}


TEST_F(CompositeDistCompositionTest, cdf) {
    int Ntest = 100;
    for(int n=0;n<Ntest;n++)
    {
        auto v = this->cd.sample(env->get_rng());
        double cdf = this->cd.cdf(v);
        double cdf2=1.0;
        cdf2*=dist0.cdf(v[0]);
        cdf2*=dist1.cdf(v[1]);
        cdf2*=dist2.cdf(v[2]);
        EXPECT_DOUBLE_EQ(cdf2,cdf);
        EXPECT_TRUE(std::isfinite(cdf));
        EXPECT_LE(0,cdf);
        EXPECT_LE(cdf,1);
    }
}    

TEST_F(CompositeDistCompositionTest, pdf) {
    int Ntest = 100;
    for(int n=0;n<Ntest;n++)
    {
        auto v = this->cd.sample(env->get_rng());
        double pdf = this->cd.pdf(v);
        double pdf2=1.0;
        pdf2*=dist0.pdf(v[0]);
        pdf2*=dist1.pdf(v[1]);
        pdf2*=dist2.pdf(v[2]);
        EXPECT_DOUBLE_EQ(pdf2,pdf);
        EXPECT_TRUE(std::isfinite(pdf));
        EXPECT_LE(0,pdf);
    }
}    

TEST_F(CompositeDistCompositionTest, llh) {
    int Ntest = 100;
    for(int n=0;n<Ntest;n++)
    {
        auto v = this->cd.sample(env->get_rng());
        double llh = this->cd.llh(v);
        double llh2=0;
        llh2+=dist0.llh(v[0]);
        llh2+=dist1.llh(v[1]);
        llh2+=dist2.llh(v[2]);
        EXPECT_DOUBLE_EQ(llh2,llh);
        EXPECT_TRUE(std::isfinite(llh));
    }
}    
