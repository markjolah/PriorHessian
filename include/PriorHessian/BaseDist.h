/** @file BaseDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The Base classes for UnivariateDist and MultivariateDist.
 * 
 * Distinguishing univariate from multivariate distributions is important as the types
 * of most functions are different (scalar vs vector).  For maximum efficiency we wish for
 * scalar types to not be forced to used size=1 vectors, but instead call directly with double
 * arguments.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_BASEDIST_H
#define _PRIOR_HESSIAN_BASEDIST_H

#include<string>
#include<vector>
#include<utility>
#include<algorithm>
#include<typeindex>

#include<armadillo>

#include<BacktraceException/BacktraceException.h>

namespace prior_hessian {
 
using IdxT = arma::uword;
using UVecT = arma::Col<IdxT>;
using VecT = arma::Col<double>;
using MatT = arma::Mat<double>;
using StringVecT = std::vector<std::string>; 
using TypeInfoVecT = std::vector<std::type_index>;

using UniformDistT = std::uniform_real_distribution<double>;

using PriorHessianError = backtrace_exception::BacktraceException;

/** @brief Indicates a index access was out of bounds
 */
struct IndexError : public PriorHessianError 
{
    IndexError(std::string message) : PriorHessianError("IndexError",message) {}
};

struct ParameterValueError : public PriorHessianError 
{
    ParameterValueError(std::string message) : PriorHessianError("ParameterValueError",message) {}
};

struct ParameterNameError : public PriorHessianError 
{
    ParameterNameError(std::string message) : PriorHessianError("ParameterNameError",message) {}
};

struct RuntimeTypeError : public PriorHessianError 
{
    RuntimeTypeError(std::string message) : PriorHessianError("RuntimeTypeError",message) {}
};

struct NumericalOverflowError : public PriorHessianError 
{
    NumericalOverflowError(std::string message) : PriorHessianError("NumericalOverflowError",message) {}
};

template<class T>
T square(T t) 
{ 
    return t*t;
}

//Forward decl
template<class RngT>
class CompositeDist;

/** @brief Encapsulates the common functionality of UnivariateDist and MultivariateDist
 */
class BaseDist
{
 template<class RngT> friend class CompositeDist;
public:
    BaseDist(StringVecT &&param_names);
    const StringVecT& param_names() const;
    IdxT num_params() const;    
    void set_param_names(const StringVecT& desc); 
protected:
    template<class IterT> void append_param_names(IterT& p) const;
    template<class IterT> void set_param_names(IterT& d); 
    StringVecT _param_names;
};

/** @brief Class templates to utilize sequencing behaviour of std::initializer_list expressions.
 * 
 * These class templates are intended to be used in variadic template functions to sequence the order of calls as
 * a std::initializer_list.
 * 
 */
namespace meta {
    //inline void call_in_order() { }
    template<class T>
    void call_in_order(std::initializer_list<T>) 
    { }

    template<class T>
    constexpr T sum_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{0},std::plus<T>()); }
    
    template<class T>
    constexpr T prod_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{1},std::multiplies<T>()); }

    
    constexpr IdxT unordered_sum() { return 0;}
    
    template<class T>
    constexpr T unordered_sum(T i) { return i;}
    
    template<class T, class... Ts>
    constexpr auto unordered_sum(T i,Ts... args) 
    { return i + unordered_sum(args...);}
}

inline
BaseDist::BaseDist(StringVecT &&param_names) : 
    _param_names(std::move(param_names)) 
{ }

inline
const StringVecT& BaseDist::param_names() const 
{ return _param_names; }

inline
IdxT BaseDist::num_params() const 
{ return _param_names.size(); }

template<class IterT>
void BaseDist::append_param_names(IterT& p) const 
{ 
    p = std::copy(_param_names.cbegin(),_param_names.cend(),p); //Make sure to update p to new position
} 

inline
void BaseDist::set_param_names(const StringVecT& desc) 
{ 
    auto iter = desc.cbegin();
    set_param_names(iter);
}

template<class IterT>
void BaseDist::set_param_names(IterT& d) 
{ for(IdxT i=0; i<num_params(); i++) _param_names[i] = *d++; }



} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_BASEDIST_H */
