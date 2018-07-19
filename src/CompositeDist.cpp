/** @file CompositeDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief CompositeDist and associated classes and nested classes
 * 
 * 
 */

#include "PriorHessian/CompositeDist.h"

namespace prior_hessian {

CompositeDist::CompositeDist() 
    : handle{new EmptyDistTuple{}}
{ initialize_from_handle(); }
    
CompositeDist::CompositeDist(const CompositeDist &o) 
    : handle{o.handle->clone()}
{ initialize_from_handle(); }
    
CompositeDist::CompositeDist(CompositeDist &&o) 
    : handle{std::move(o.handle)},
      param_name_idx{std::move(o.param_name_idx)}
{ }

void CompositeDist::clear()
{
    handle = std::unique_ptr<DistTupleHandle>{new EmptyDistTuple{}};
    initialize_from_handle();
}
   
CompositeDist& CompositeDist::operator=(const CompositeDist &o) 
{
    if(this == &o) return *this; //Ignore self-assignment
    handle = o.handle->clone();
    initialize_from_handle();
    return *this;
}

CompositeDist& CompositeDist::operator=(CompositeDist &&o) 
{
    if(this == &o) return *this; //Ignore self-assignment
    handle = std::move(o.handle);
    param_name_idx = std::move(o.param_name_idx);
    return *this;
}

bool CompositeDist::operator==(const CompositeDist &o) const 
{
    if(is_empty()) return o.is_empty(); //Empty classes are equal to themselves
    if(o.is_empty()) return false;
    auto &h1 = *handle;
    auto &h2 = *o.handle;
    if(typeid(h1) != typeid(h2)) return false;
    return h1.is_equal(h2);
}

std::ostream& operator<<(std::ostream &out,const CompositeDist &comp_dist)
{
    out<<"[CompositeDist]:\n";
    out<<"  NumComponentDists:"<<comp_dist.num_component_dists()<<"\n";
    out<<"  NumDim:"<<comp_dist.num_dim()<<"\n";
    out<<"  ComponentNumDim:"<<comp_dist.components_num_dim().t();
    out<<"  LBound:"<<comp_dist.lbound().t();
    out<<"  UBound:"<<comp_dist.ubound().t();
    auto vars=comp_dist.dim_variables();
    out<<"  Vars:[";
    for(auto v: vars) out<<v<<",";
    out<<"]\n";

    out<<"  NumParams:"<<comp_dist.num_params()<<"\n";
    out<<"  ComponentNumParams:"<<comp_dist.components_num_params().t();
    out<<"  Params:"<<comp_dist.params().t();
    auto param_names=comp_dist.param_names();
    out<<"  ParamDesc:[";
    for(auto &v: param_names) out<<v<<",";
    out<<"]\n";
    
    return out;
}

bool CompositeDist::has_param(const std::string &name) const
{
    return param_name_idx.find(name) != param_name_idx.end();
}

double CompositeDist::get_param_value(const std::string &name) const
{
    auto it = param_name_idx.find(name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named:"<<name;
        throw ParameterNameError(msg.str());
    }
    return params()[it->second];
}

int CompositeDist::get_param_index(const std::string &name) const
{
    auto it = param_name_idx.find(name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named:"<<name;
        throw ParameterNameError(msg.str());
    }
    return it->second;
}

void CompositeDist::set_param_value(const std::string &name, double value)
{
    auto it = param_name_idx.find(name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named:"<<name;
        throw ParameterNameError(msg.str());
    }
    auto ps = params();
    ps[it->second]=value;
    set_params(ps);
}

//Called on every new initialization
void CompositeDist::initialize_from_handle()
{
    param_name_idx = initialize_param_name_idx(param_names());
}

CompositeDist::ParamNameMapT 
CompositeDist::initialize_param_name_idx(const StringVecT &names)
{
    ParamNameMapT name_idx;
    for(IdxT i=0; i<names.size(); i++) name_idx[names[i]] = i;
    if(name_idx.size() < names.size()){
        std::ostringstream msg;
        msg<<"Parameter names contain duplicate values. Got: "<<names.size()<<" name, but only "<<name_idx.size()<<" are unique. Names:";
        for(auto n: names) msg<<n<<",";
        msg<<"]";
        throw ParameterNameUniquenessError(msg.str());
    }
    return name_idx;
}

} /* namespace prior_hessian */
