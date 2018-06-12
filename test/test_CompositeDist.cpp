/** @file test_CompositeDists.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Use googletest to test the CompositeDist 
 */
#include <cmath>
#include "gtest/gtest.h"

#include "test_prior_hessian.h"

TEST_F(CompositeDistCompositionTest, copy_construction) {
    DistT &cd = this->cd;
    IdxT Nparams = cd.num_params();
    IdxT Ndim = cd.num_dim();
    auto params = cd.params();
    auto lbound = cd.lbound();
    DistT dist_copy{this->cd};  //copy construct
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
    env->reset_rng();
    auto v = cd.sample(env->get_rng());
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    ASSERT_EQ(v2.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(v[i],v2[i]);
}

TEST_F(CompositeDistCompositionTest, copy_assignment) {
    DistT &cd = this->cd;
    DistT dist_copy{};
    volatile double _foo;
    _foo = (dist_copy.num_dim(),_foo); //Force something to happen with dist_copy first
    
    dist_copy=this->cd;  //Move construct
    EXPECT_EQ(dist_copy.num_dim(), cd.num_dim());
    EXPECT_EQ(dist_copy.num_params(), cd.num_params());
    //Check copy of parameters is successful
    auto params = cd.params();
    auto lbound = cd.lbound();
    IdxT Nparams = cd.num_params();
    IdxT Ndim = cd.num_dim();
    auto params_copy = dist_copy.params();
    ASSERT_EQ(params_copy.n_elem, Nparams);
    for(IdxT i=0; i < Nparams; i++)
        EXPECT_EQ(params[i], params_copy[i]);
    auto lbound_copy = dist_copy.lbound();
    ASSERT_EQ(lbound_copy.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(lbound[i], lbound_copy[i])<<i;
    //Check repeatability of rng generation
    env->reset_rng();
    auto v = cd.sample(env->get_rng());
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    ASSERT_EQ(v2.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(v[i],v2[i]);
}


TEST_F(CompositeDistCompositionTest, move_construction) {
    DistT &cd = this->cd;
    auto params = cd.params();
    auto lbound = cd.lbound();
    env->reset_rng();
    auto v = cd.sample(env->get_rng());
    IdxT Nparams = cd.num_params();
    IdxT Ndim = cd.num_dim();
    DistT dist_copy{std::move(this->cd)};  //Move construct
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
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    ASSERT_EQ(v2.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(v[i],v2[i]);
}

TEST_F(CompositeDistCompositionTest, move_assignment) {
    DistT &cd = this->cd;
    DistT dist_copy{std::make_tuple(prior_hessian::NormalDist{})}; //Make something useful to force compiler to do something.
    auto params = cd.params();
    auto lbound = cd.lbound();
    env->reset_rng();
    auto v = cd.sample(env->get_rng());
    IdxT Nparams = cd.num_params();
    IdxT Ndim = cd.num_dim();
    dist_copy = std::move(this->cd); //Now move over it with our test fixture dist
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
    env->reset_rng();
    auto v2 = dist_copy.sample(env->get_rng());
    ASSERT_EQ(v2.n_elem, Ndim);
    for(IdxT i=0; i < Ndim; i++)
        EXPECT_EQ(v[i],v2[i]);
}

TEST_F(CompositeDistCompositionTest, num_component_dists) {
    DistT &cd = this->cd;
    EXPECT_EQ(cd.num_component_dists(),3);
}

TEST_F(CompositeDistCompositionTest, component_types) {
    DistT &cd = this->cd;
    auto types = cd.component_types();
    ASSERT_EQ(cd.num_component_dists(), types.size());
    EXPECT_EQ(std::type_index(typeid(dist0)), types[0]);
    EXPECT_EQ(std::type_index(typeid(dist1)), types[1]);
    EXPECT_EQ(std::type_index(typeid(dist2)), types[2]);
}

TEST_F(CompositeDistCompositionTest, num_dim) {
    DistT &cd = this->cd;
    EXPECT_EQ(cd.num_dim(),3);
}

TEST_F(CompositeDistCompositionTest, components_num_dim) {
    DistT &cd = this->cd;
    auto ndim = cd.components_num_dim();
    ASSERT_EQ(cd.num_component_dists(), ndim.size());
    EXPECT_EQ(dist0.num_dim(), ndim[0]);
    EXPECT_EQ(dist1.num_dim(), ndim[1]);
    EXPECT_EQ(dist2.num_dim(), ndim[2]);
}

TEST_F(CompositeDistCompositionTest, dim_variables) {
    DistT &cd = this->cd;
    auto vars = cd.dim_variables();
    ASSERT_EQ(cd.num_dim(), vars.size());
    EXPECT_EQ(dist0.var_name(), vars[0]);
    EXPECT_EQ(dist1.var_name(), vars[1]);
    EXPECT_EQ(dist2.var_name(), vars[2]);
}

TEST_F(CompositeDistCompositionTest, set_dim_variables) {
    DistT &cd = this->cd;
    auto vars = cd.dim_variables();
    for(auto& v:vars) v.append("foo");
    cd.set_dim_variables(vars);
    auto vars2 = cd.dim_variables();
    ASSERT_EQ(cd.num_dim(), vars2.size());
    EXPECT_EQ(dist0.var_name()+"foo", vars2[0]);
    EXPECT_EQ(dist1.var_name()+"foo", vars2[1]);
    EXPECT_EQ(dist2.var_name()+"foo", vars2[2]);
}

TEST_F(CompositeDistCompositionTest, lbound) {
    DistT &cd = this->cd;
    auto lbound = cd.lbound();
    ASSERT_EQ(cd.num_dim(), lbound.size());
    EXPECT_EQ(dist0.lbound(), lbound[0]);
    EXPECT_EQ(dist1.lbound(), lbound[1]);
    EXPECT_EQ(dist2.lbound(), lbound[2]);
}

TEST_F(CompositeDistCompositionTest, set_lbound) {
    DistT &cd = this->cd;
    auto lbound = cd.lbound();
    for(IdxT i=0;i<lbound.size();i++) lbound[i]=3.141+i;
    cd.set_lbound(lbound);
    auto lbound2 = cd.lbound();
    ASSERT_EQ(cd.num_dim(), lbound2.size());
    EXPECT_EQ(3.141+0, lbound2[0]);
    EXPECT_EQ(3.141+1, lbound2[1]);
    EXPECT_EQ(3.141+2, lbound2[2]);
}

TEST_F(CompositeDistCompositionTest, ubound) {
    DistT &cd = this->cd;
    auto ubound = cd.ubound();
    ASSERT_EQ(cd.num_dim(), ubound.size());
    EXPECT_EQ(dist0.ubound(), ubound[0]);
    EXPECT_EQ(dist1.ubound(), ubound[1]);
    EXPECT_EQ(dist2.ubound(), ubound[2]);
}

TEST_F(CompositeDistCompositionTest, set_ubound) {
    DistT &cd = this->cd;
    auto ubound = cd.ubound();
    for(IdxT i=0;i<ubound.size();i++) ubound[i]=3.141+i;
    cd.set_ubound(ubound);
    auto ubound2 = cd.ubound();
    ASSERT_EQ(cd.num_dim(), ubound2.size());
    EXPECT_EQ(3.141+0, ubound2[0]);
    EXPECT_EQ(3.141+1, ubound2[1]);
    EXPECT_EQ(3.141+2, ubound2[2]);
}

TEST_F(CompositeDistCompositionTest, set_bounds) {
    DistT &cd = this->cd;
    auto lbounds = cd.lbound();
    auto ubounds = cd.ubound();
    for(IdxT i=0;i<lbounds.size();i++) {lbounds[i]=3.141+i; ubounds[i]=i+10;}
    cd.set_bounds(lbounds,ubounds);
    auto lbounds2 = cd.lbound();
    auto ubounds2 = cd.ubound();
    ASSERT_EQ(cd.num_dim(), ubounds2.size());
    ASSERT_EQ(cd.num_dim(), lbounds2.size());
    EXPECT_EQ(3.141+0, lbounds2[0]);
    EXPECT_EQ(3.141+1, lbounds2[1]);
    EXPECT_EQ(3.141+2, lbounds2[2]);
    EXPECT_EQ(10+0, ubounds2[0]);
    EXPECT_EQ(10+1, ubounds2[1]);
    EXPECT_EQ(10+2, ubounds2[2]);
}

TEST_F(CompositeDistCompositionTest, component_num_params) {
    DistT &cd = this->cd;
    auto nps = cd.components_num_params();
    ASSERT_EQ(cd.num_params(), arma::sum(nps));
    EXPECT_EQ(dist0.num_params(), nps[0]);
    EXPECT_EQ(dist1.num_params(), nps[1]);
    EXPECT_EQ(dist2.num_params(), nps[2]);
}

TEST_F(CompositeDistCompositionTest, params) {
    DistT &cd = this->cd;
    auto params = cd.params();
    auto nps = cd.components_num_params();
    ASSERT_EQ(cd.num_params(), params.size());
    ASSERT_EQ(cd.num_component_dists(), nps.size());
    IdxT j=0;
    for(IdxT k=0; k<nps[0]; k++)
        EXPECT_EQ(dist0.get_param(k),params[j++]);
    for(IdxT k=0; k<nps[1]; k++)
        EXPECT_EQ(dist1.get_param(k),params[j++]);
    for(IdxT k=0; k<nps[2]; k++)
        EXPECT_EQ(dist2.get_param(k),params[j++]);
}

TEST_F(CompositeDistCompositionTest, set_params) {
    DistT &cd = this->cd;
    auto params = cd.params();
    for(auto& p:params) p*=1.01;
    cd.set_params(params);
    auto params2 = cd.params();
    ASSERT_EQ(cd.num_params(), params2.size());
    for(IdxT k=0;k<params2.size();k++) EXPECT_EQ(params[k],params2[k]);
}

TEST_F(CompositeDistCompositionTest, param_names) {
    DistT &cd = this->cd;
    auto pd = cd.param_names();
    auto nps = cd.components_num_params();
    ASSERT_EQ(cd.num_params(), pd.size());
    ASSERT_EQ(cd.num_component_dists(), nps.size());
    IdxT j=0;
    for(IdxT k=0; k<nps[0]; k++)
        EXPECT_EQ(dist0.param_names()[k],pd[j++]);
    for(IdxT k=0; k<nps[1]; k++)
        EXPECT_EQ(dist1.param_names()[k],pd[j++]);
    for(IdxT k=0; k<nps[2]; k++)
        EXPECT_EQ(dist2.param_names()[k],pd[j++]);
}

TEST_F(CompositeDistCompositionTest, set_param_names) {
    DistT &cd = this->cd;
    auto pd = cd.param_names();
    ASSERT_EQ(cd.num_params(), pd.size());
    for(auto& p:pd) p.append("abc1234567890");    
    cd.set_param_names(pd);
    auto pd2 = cd.param_names();
    ASSERT_EQ(cd.num_params(), pd2.size());
    for(IdxT k=0;k<pd2.size();k++) 
        EXPECT_EQ(pd[k],pd2[k]);
    auto vals = cd.params();
    for(IdxT k=0;k<cd.num_params();k++)
        EXPECT_EQ(vals[k],cd.get_param_value(pd2[k])); //Check the parameter index map still works
    pd[0] = pd[1];
    EXPECT_THROW(cd.set_param_names(pd), prior_hessian::ParameterNameUniquenessError);  //already used name
    

}

TEST_F(CompositeDistCompositionTest, has_param) {
    DistT &cd = this->cd;
    auto names = cd.param_names();
    for(auto n: names) EXPECT_TRUE(cd.has_param(n));
    EXPECT_FALSE(cd.has_param(""));
    EXPECT_FALSE(cd.has_param("foo"));
}

TEST_F(CompositeDistCompositionTest, get_param_value) {
    DistT &cd = this->cd;
    auto names = cd.param_names();
    auto vals = cd.params();
    for(IdxT i=0;i<cd.num_params();i++) EXPECT_EQ(cd.get_param_value(names[i]),vals[i]);
    EXPECT_THROW(cd.get_param_value(""),prior_hessian::ParameterNameError);
    EXPECT_THROW(cd.get_param_value("foo"),prior_hessian::ParameterNameError);
}

TEST_F(CompositeDistCompositionTest, set_param_value) {
    DistT &cd = this->cd;
    auto names = cd.param_names();
    auto vals = cd.params();
    auto new_vals = vals+0.001;
    for(IdxT i=0;i<cd.num_params();i++) {
        EXPECT_EQ(cd.get_param_value(names[i]),vals[i]);
        cd.set_param_value(names[i],new_vals[i]);
        EXPECT_EQ(cd.get_param_value(names[i]),new_vals[i]);
    }
    EXPECT_THROW(cd.set_param_value("",17),prior_hessian::ParameterNameError);
    EXPECT_THROW(cd.set_param_value("foo",21),prior_hessian::ParameterNameError);
}

TEST_F(CompositeDistCompositionTest, rename_param) {
    DistT &cd = this->cd;
    auto names = cd.param_names();
    auto vals = cd.params();
    for(IdxT i=0;i<cd.num_params();i++) {
        EXPECT_EQ(cd.get_param_value(names[i]),vals[i]);
        if(i<cd.num_params()-1) {
            EXPECT_THROW(cd.rename_param(names[i],names[i+1]),prior_hessian::ParameterNameUniquenessError);  //already used name
        }
        EXPECT_EQ(cd.get_param_value(names[i]),vals[i]);
        
        cd.rename_param(names[i],names[i]+"'");
        auto new_names = cd.param_names();
        EXPECT_TRUE(cd.has_param(new_names[0]));
        EXPECT_TRUE(cd.has_param(names[i]+"'"));
        EXPECT_EQ(cd.get_param_value(names[i]+"'"),vals[i]);
        EXPECT_THROW(cd.get_param_value(names[i]),prior_hessian::ParameterNameError);
        EXPECT_THROW(cd.rename_param(names[i],names[i]+"'"),prior_hessian::ParameterNameError);
    }
    EXPECT_THROW(cd.rename_param("",""),prior_hessian::ParameterNameError);
    EXPECT_THROW(cd.rename_param("foo","bar"),prior_hessian::ParameterNameError);
}

TEST_F(CompositeDistCompositionTest, cdf) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        double cdf = cd.cdf(v);
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
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        double pdf = cd.pdf(v);
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
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        double llh = cd.llh(v);
        double llh2=0;
        llh2+=dist0.llh(v[0]);
        llh2+=dist1.llh(v[1]);
        llh2+=dist2.llh(v[2]);
        EXPECT_DOUBLE_EQ(llh2,llh);
        EXPECT_TRUE(std::isfinite(llh));
    }
}    

TEST_F(CompositeDistCompositionTest, rllh) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        double rllh = cd.rllh(v);
        double rllh2=0;
        rllh2 += dist0.rllh(v[0]);
        rllh2 += dist1.rllh(v[1]);
        rllh2 += dist2.rllh(v[2]);
        EXPECT_DOUBLE_EQ(rllh2,rllh);
        EXPECT_TRUE(std::isfinite(rllh));
    }
}    

TEST_F(CompositeDistCompositionTest, grad) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad = cd.grad(v);
        ASSERT_EQ(grad.n_elem, cd.num_dim());
        EXPECT_EQ(dist0.grad(v[0]), grad[0]);
        EXPECT_EQ(dist1.grad(v[1]), grad[1]);
        EXPECT_EQ(dist2.grad(v[2]), grad[2]);
    }
}

TEST_F(CompositeDistCompositionTest, grad2) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad2 = cd.grad2(v);
        ASSERT_EQ(grad2.n_elem, cd.num_dim());
        EXPECT_EQ(dist0.grad2(v[0]), grad2[0]);
        EXPECT_EQ(dist1.grad2(v[1]), grad2[1]);
        EXPECT_EQ(dist2.grad2(v[2]), grad2[2]);
    }
}

TEST_F(CompositeDistCompositionTest, hess) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto hess = cd.hess(v);
        ASSERT_EQ(hess.n_rows, cd.num_dim());
        ASSERT_EQ(hess.n_cols, cd.num_dim());
        EXPECT_EQ(dist0.grad2(v[0]), hess(0,0));
        EXPECT_EQ(dist1.grad2(v[1]), hess(1,1));
        EXPECT_EQ(dist2.grad2(v[2]), hess(2,2));
    }
}

TEST_F(CompositeDistCompositionTest, grad_accumulate) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad = cd.make_zero_grad();
        ASSERT_EQ(grad.n_elem, cd.num_dim());
        cd.grad_accumulate(v,grad);
        EXPECT_EQ(dist0.grad(v[0]), grad[0]);
        EXPECT_EQ(dist1.grad(v[1]), grad[1]);
        EXPECT_EQ(dist2.grad(v[2]), grad[2]);
    }
}

TEST_F(CompositeDistCompositionTest, grad2_accumulate) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad2 = cd.make_zero_grad();
        ASSERT_EQ(grad2.n_elem, cd.num_dim());
        cd.grad2_accumulate(v,grad2);
        EXPECT_EQ(dist0.grad2(v[0]), grad2[0]);
        EXPECT_EQ(dist1.grad2(v[1]), grad2[1]);
        EXPECT_EQ(dist2.grad2(v[2]), grad2[2]);
    }
}

TEST_F(CompositeDistCompositionTest, hess_accumulate) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto hess = cd.make_zero_hess();
        ASSERT_EQ(hess.n_rows, cd.num_dim());
        ASSERT_EQ(hess.n_cols, cd.num_dim());
        cd.hess_accumulate(v,hess);
        EXPECT_EQ(dist0.grad2(v[0]), hess(0,0));
        EXPECT_EQ(dist1.grad2(v[1]), hess(1,1));
        EXPECT_EQ(dist2.grad2(v[2]), hess(2,2));
    }
}

TEST_F(CompositeDistCompositionTest, grad_grad2_accumulate) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad = cd.make_zero_grad();
        auto grad2 = cd.make_zero_grad();
        ASSERT_EQ(grad.n_elem, cd.num_dim());
        ASSERT_EQ(grad2.n_elem, cd.num_dim());
        cd.grad_grad2_accumulate(v,grad,grad2);
        EXPECT_DOUBLE_EQ(dist0.grad(v[0]), grad[0]);
        EXPECT_DOUBLE_EQ(dist1.grad(v[1]), grad[1]);
        EXPECT_DOUBLE_EQ(dist2.grad(v[2]), grad[2]);
        EXPECT_DOUBLE_EQ(dist0.grad2(v[0]), grad2[0]);
        EXPECT_DOUBLE_EQ(dist1.grad2(v[1]), grad2[1]);
        EXPECT_DOUBLE_EQ(dist2.grad2(v[2]), grad2[2]);
   }
}

TEST_F(CompositeDistCompositionTest, grad_hess_accumulate) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad = cd.make_zero_grad();
        ASSERT_EQ(grad.n_elem, cd.num_dim());
        auto hess = cd.make_zero_hess();
        ASSERT_EQ(hess.n_rows, cd.num_dim());
        ASSERT_EQ(hess.n_cols, cd.num_dim());
        cd.grad_hess_accumulate(v,grad,hess);
        EXPECT_DOUBLE_EQ(dist0.grad(v[0]), grad[0]);
        EXPECT_DOUBLE_EQ(dist1.grad(v[1]), grad[1]);
        EXPECT_DOUBLE_EQ(dist2.grad(v[2]), grad[2]);
        EXPECT_DOUBLE_EQ(dist0.grad2(v[0]), hess(0,0));
        EXPECT_DOUBLE_EQ(dist1.grad2(v[1]), hess(1,1));
        EXPECT_DOUBLE_EQ(dist2.grad2(v[2]), hess(2,2));
    }
}

TEST_F(CompositeDistCompositionTest, g) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto grad = cd.make_zero_grad();
        ASSERT_EQ(grad.n_elem, cd.num_dim());
        auto hess = cd.make_zero_hess();
        ASSERT_EQ(hess.n_rows, cd.num_dim());
        ASSERT_EQ(hess.n_cols, cd.num_dim());
        cd.grad_hess_accumulate(v,grad,hess);
        EXPECT_DOUBLE_EQ(dist0.grad(v[0]), grad[0]);
        EXPECT_DOUBLE_EQ(dist1.grad(v[1]), grad[1]);
        EXPECT_DOUBLE_EQ(dist2.grad(v[2]), grad[2]);
        EXPECT_DOUBLE_EQ(dist0.grad2(v[0]), hess(0,0));
        EXPECT_DOUBLE_EQ(dist1.grad2(v[1]), hess(1,1));
        EXPECT_DOUBLE_EQ(dist2.grad2(v[2]), hess(2,2));
    }
}

TEST_F(CompositeDistCompositionTest, sample_bounds_test) {
    prior_hessian::VecT lbound={0.2,0.3,0.4};
    prior_hessian::VecT ubound={1.2,1.3,1.4};
    cd.set_bounds(lbound,ubound);
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        for(prior_hessian::IdxT k=0;k<cd.num_dim();k++){
            EXPECT_LE(lbound[k],v[k]);
            EXPECT_GE(ubound[k],v[k]);
        }
    }
}

TEST_F(CompositeDistCompositionTest, sample_vector_repeatability) {
    env->reset_rng();
    int N = this->Ntest;
    auto sample = cd.sample(env->get_rng(),N);
    ASSERT_EQ(sample.n_rows,cd.num_dim());
    ASSERT_EQ(sample.n_cols,N);
    env->reset_rng();
    for(int n=0; n < N; n++)
    {
        auto v = cd.sample(env->get_rng());
        ASSERT_EQ(v.n_elem,cd.num_dim());
        for(prior_hessian::IdxT k=0; k<cd.num_dim(); k++){
            EXPECT_EQ(sample(k,n),v[k]);
        }
    }
}

TEST_F(CompositeDistCompositionTest, llh_components) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto llh_comp = cd.llh_components(v);
        ASSERT_EQ(llh_comp.n_elem, cd.num_dim());
        EXPECT_EQ(dist0.llh(v[0]), llh_comp[0]);
        EXPECT_EQ(dist1.llh(v[1]), llh_comp[1]);
        EXPECT_EQ(dist2.llh(v[2]), llh_comp[2]);
    }
}

TEST_F(CompositeDistCompositionTest, rllh_components) {
    for(int n=0; n < this->Ntest; n++)
    {
        auto v = cd.sample(env->get_rng());
        auto rllh_comp = cd.rllh_components(v);
        ASSERT_EQ(rllh_comp.n_elem, cd.num_dim());
        EXPECT_EQ(dist0.rllh(v[0]), rllh_comp[0]);
        EXPECT_EQ(dist1.rllh(v[1]), rllh_comp[1]);
        EXPECT_EQ(dist2.rllh(v[2]), rllh_comp[2]);
    }
}
