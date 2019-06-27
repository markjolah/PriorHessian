/** @file CompositeDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
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
    : handle{o.handle->clone()},
      component_names_initialized(o.component_names_initialized),
      dim_variables_initialized(o.dim_variables_initialized),
      param_names_initialized(o.param_names_initialized)
{
    if(component_names_initialized) _component_names = o._component_names;
    if(dim_variables_initialized) _dim_variables = o._dim_variables;
    if(param_names_initialized) _param_names = o._param_names;
}
    
CompositeDist::CompositeDist(CompositeDist &&o) 
    : handle{std::move(o.handle)},
      component_names_initialized(o.component_names_initialized),
      dim_variables_initialized(o.dim_variables_initialized),
      param_names_initialized(o.param_names_initialized)
{
    if(component_names_initialized) _component_names = std::move(o._component_names);
    if(dim_variables_initialized) _dim_variables = std::move(o._dim_variables);
    if(param_names_initialized) _param_names = std::move(o._param_names);
}
void CompositeDist::clear()
{
    handle = std::unique_ptr<DistTupleHandle>{new EmptyDistTuple{}};
    initialize_from_handle();
}
   
CompositeDist& CompositeDist::operator=(const CompositeDist &o) 
{
    if(this == &o) return *this; //Ignore self-assignment
    handle = o.handle->clone();
    component_names_initialized = o.component_names_initialized;
    dim_variables_initialized = o.dim_variables_initialized;
    param_names_initialized = o.param_names_initialized;
    if(component_names_initialized) _component_names = o._component_names;
    if(dim_variables_initialized) _dim_variables = o._dim_variables;
    if(param_names_initialized) _param_names = o._param_names;
    return *this;
}

CompositeDist& CompositeDist::operator=(CompositeDist &&o) 
{
    if(this == &o) return *this; //Ignore self-assignment
    handle = std::move(o.handle);
    component_names_initialized = o.component_names_initialized;
    dim_variables_initialized = o.dim_variables_initialized;
    param_names_initialized = o.param_names_initialized;
    if(component_names_initialized) _component_names = std::move(o._component_names);
    if(dim_variables_initialized) _dim_variables = std::move(o._dim_variables);
    if(param_names_initialized) _param_names = std::move(o._param_names);
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
    out<<"  NumComponentDists:"<<comp_dist.num_components()<<"\n";
    out<<"  ComponentNames:[";
    for(auto v: comp_dist.component_names()) out<<v<<",";
    out<<"]\n";
    out<<"  NumDim:"<<comp_dist.num_dim()<<"\n";
    out<<"  ComponentNumDim:"<<comp_dist.num_dim_components().t();
    out<<"  LBound:"<<comp_dist.lbound().t();
    out<<"  UBound:"<<comp_dist.ubound().t();
    out<<"  GlobalLBound:"<<comp_dist.global_lbound().t();
    out<<"  GlobalUBound:"<<comp_dist.global_ubound().t();
    out<<"  DimVars:[";
    for(auto v: comp_dist.dim_variables()) out<<v<<",";
    out<<"]\n";
    out<<"  NumParams:"<<comp_dist.num_params()<<"\n";
    out<<"  ComponentNumParams:"<<comp_dist.num_params_components().t();
    out<<"  Params:"<<comp_dist.params().t();
    out<<"  ParamsLbound:"<<comp_dist.params_lbound().t();
    out<<"  ParamsUbound:"<<comp_dist.params_ubound().t();
    out<<"  ParamNames:[";
    for(auto &v: comp_dist.param_names()) out<<v<<",";
    out<<"]\n";
    return out;
}


const StringVecT& CompositeDist::component_names() const
{
    if(!component_names_initialized) initialize_component_names();
    return _component_names;
}

const StringVecT& CompositeDist::dim_variables() const
{
    if(!dim_variables_initialized) initialize_dim_variables();
    return _dim_variables;
}

const StringVecT& CompositeDist::param_names() const
{
    if(!param_names_initialized) initialize_param_names();
    return _param_names;
}

void CompositeDist::initialize_component_names() const
{
    _component_names.clear();
    for(IdxT n=0; n<num_components(); n++) {
        std::ostringstream name;
        name<<"D"<<n+1;
        _component_names.emplace_back(name.str());
    }
    component_names_initialized = true;
}

void CompositeDist::initialize_dim_variables() const
{
    _dim_variables.clear();
    for(IdxT n=0; n<num_dim(); n++) {
        std::ostringstream name;
        name<<"v"<<n+1;
        _dim_variables.emplace_back(name.str());
    }
    dim_name_idx = initialize_name_idx(_param_names);
    dim_variables_initialized = true;
}

void CompositeDist::initialize_param_names() const
{
    _param_names.clear();
    const auto dist_ndim = num_params_components();
    auto dist_names = component_names();
    auto param_names = handle->param_names();
    auto param_names_iter = param_names.begin();
    for(IdxT n=0; n<num_components(); n++) { 
        for(IdxT k=0; k<dist_ndim[n]; k++) {
            std::ostringstream name;
            name<<dist_names[n] << "_" << *param_names_iter++;
            _param_names.emplace_back(name.str());
        }
    }
    param_name_idx = initialize_name_idx(_param_names);
    param_names_initialized = true;
}



bool CompositeDist::has_dim_variable(const std::string &name) const
{
    if(!dim_variables_initialized) initialize_dim_variables();
    return dim_name_idx.find(name) != dim_name_idx.end();
}

IdxT CompositeDist::get_dim_variable_index(const std::string &name) const
{
    if(!dim_variables_initialized) initialize_dim_variables();
    auto it = dim_name_idx.find(name);
    if(it == dim_name_idx.end()) {
        std::ostringstream msg;
        msg << "No dimension variable found named: "<<name;
        throw ParameterNameError(msg.str());
    }
    return it->second;
}

void CompositeDist::rename_dim_variable(const std::string &old_name,std::string new_name)
{
    if(!dim_names_initialized) initialize_dim_variables();
    auto it = dim_name_idx.find(old_name);
    if(it == dim_name_idx.end()) {
        std::ostringstream msg;
        msg << "No dimension variable found named:"<<old_name;
        throw ParameterNameError(msg.str());
    }
    _dim_variables[it->second] = new_name;
    //update name index
    _dim_name_idx.erase(_dim_variables[it->second]);
    _dim_name_idx[std::move(new_name)] = it->second;
}

bool CompositeDist::has_param(const std::string &name) const
{
    if(!param_names_initialized) initialize_param_names();
    return param_name_idx.find(name) != param_name_idx.end();
}

double CompositeDist::get_param_value(const std::string &name) const
{
    if(!param_names_initialized) initialize_param_names();
    auto it = param_name_idx.find(name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named: "<<name;
        throw ParameterNameError(msg.str());
    }
    return params()[it->second];
}

IdxT CompositeDist::get_param_index(const std::string &name) const
{
    if(!param_names_initialized) initialize_param_names();
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
    if(!param_names_initialized) initialize_param_names();
    auto it = param_name_idx.find(name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named:"<<name;
        throw ParameterNameError(msg.str());
    }
    auto ps = params();
    ps[it->second] = value;
    set_params(ps);
}

void CompositeDist::rename_param(const std::string &old_name,std::string new_name)
{
    if(!param_names_initialized) initialize_param_names();
    auto it = param_name_idx.find(old_name);
    if(it == param_name_idx.end()) {
        std::ostringstream msg;
        msg << "No parameter found named:"<<old_name;
        throw ParameterNameError(msg.str());
    }
    _param_names[it->second] = new_name;
    //update name index
    _param_name_idx.erase(_param_names[it->second]);
    _param_name_idx[std::move(new_name)] = it->second;
}


//Called on every new initialization
void CompositeDist::initialize_from_handle()
{
    component_names_initialized = false;
    dim_variables_initialized = false;
    param_names_initialized = false;
}

CompositeDist::NameMapT
CompositeDist::initialize_param_name_idx(const StringVecT &names)
{
    NameMapT name_idx;
    for(IdxT i=0; i<names.size(); i++) name_idx[names[i]] = i;
    if(name_idx.size() < names.size()){
        std::ostringstream msg;
        msg<<"Names contain duplicate values. Got: "<<names.size()<<" name, but only "<<name_idx.size()<<" are unique. Names:";
        for(auto n: names) msg<<n<<",";
        msg<<"]";
        throw NameUniquenessError(msg.str());
    }
    return name_idx;
}


} /* namespace prior_hessian */
