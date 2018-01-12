/** @file BaseDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
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
#include <BacktraceException/BacktraceException.h>

namespace prior_hessian {
 
using IdxT = arma::uword;
using UVecT = arma::Col<IdxT>;
using VecT = arma::Col<double>;
using MatT = arma::Mat<double>;
using StringVecT = std::vector<std::string>; 
using TypeInfoVecT = std::vector<std::type_index>;

using PriorHessianError = backtrace_exception::BacktraceException;

template<class T>
T square(T t) 
{ 
    return t*t;
}

/** @brief Encapsulates the common functionality of UnivariateDist and MultivariateDist
 */
class BaseDist
{
public:
    BaseDist(StringVecT &&params_desc);
    const StringVecT& params_desc() const;
    IdxT num_params() const;    
    template<class IterT> void insert_params_desc(IterT& p) const;
    void set_params_desc(const StringVecT& desc); 
    template<class IterT> void set_params_desc(IterT& d); 
protected:
    StringVecT _params_desc;
};

/** @brief Class templates to utilize sequencing behaviour of std::initializer_list expressions.
 * 
 * These class templates are intended to be used in variadic template functions to sequence the order of calls as
 * a std::initializer_list.
 * 
 */
namespace meta {
    template<class T>
    void call_in_order(std::initializer_list<T>) 
    { }

    template<class T>
    constexpr T sum_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{0},std::plus<T>()); }
    
    template<class T>
    constexpr T prod_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{1},std::multiplies<T>()); }

    template<class T>
    constexpr T unordered_sum(T i) { return i;}
    template<class T, class... Ts>
    constexpr auto unordered_sum(T i,Ts... args) 
    { return i + unordered_sum(args...);}
}

inline
BaseDist::BaseDist(StringVecT &&params_desc) : 
    _params_desc(std::move(params_desc)) 
{ }

inline
const StringVecT& BaseDist::params_desc() const 
{ return _params_desc; }

inline
IdxT BaseDist::num_params() const 
{ return _params_desc.size(); }

template<class IterT>
void BaseDist::insert_params_desc(IterT& p) const 
{ 
    p = std::copy(_params_desc.cbegin(),_params_desc.cend(),p); //Make sure to update p to new position
} 

inline
void BaseDist::set_params_desc(const StringVecT& desc) 
{ 
    auto iter = desc.cbegin();
    set_params_desc(iter);
}

template<class IterT>
void BaseDist::set_params_desc(IterT& d) 
{ for(IdxT i=0; i<num_params(); i++) _params_desc[i] = *d++; }



} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_BASEDIST_H */
