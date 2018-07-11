/** @file util.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 10-2017
 * @brief Utilities and namespace globals
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_UTIL_H
#define _PRIOR_HESSIAN_UTIL_H

#include<cmath>
#include<string>
#include<random>
#include<vector>
#include<typeindex>

#include<armadillo>

/**
 * 
 * 
 */
namespace prior_hessian
{

using VecT = arma::Col<double>;
using IdxT = arma::uword;
using UVecT = arma::Col<IdxT>;
using VecT = arma::Col<double>;
using MatT = arma::Mat<double>;
using StringVecT = std::vector<std::string>; 
using TypeInfoVecT = std::vector<std::type_index>;

// using UniformDistT = std::uniform_real_distribution<double>;

/* Allow easier enabale_if compilation for subclasses */
template<class T,class BaseT> 
    using EnableIfSubclassT = typename std::enable_if<std::is_base_of<std::remove_reference_t<BaseT>,std::remove_reference_t<T>>::value>::type;

template<class ReturnT, class T,class BaseT> 
    using ReturnIfSubclassT = typename std::enable_if<std::is_base_of<std::remove_reference_t<BaseT>,std::remove_reference_t<T>>::value,ReturnT>::type;


template<class T>
T square(T t) 
{ 
    return t*t;
}

// /** Compute log(1+exp(x)) 
//  */
// inline
// double log1pexp(double x) 
// {
//     if (x <= -37) return exp(x);
//     else if (x <= 18) return log1p(exp(x));
//     else if (x <= 33.3) return x + exp(-x);
//     else return x;
// }
// 
// /** Compute log(1-exp(x)) 
//  */
// inline
// double log1mexp(double x) 
// {
//     static const double ln2 = log(2);
//     if (x <= 0) return -INFINITY;
//     else if (x <= ln2) return log(-expm1(-x));
//     else return log1p(-exp(-x));
// }

} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_UTIL_H */
