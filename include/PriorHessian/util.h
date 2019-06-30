/** @file util.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief Utilities and namespace globals
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_UTIL_H
#define PRIOR_HESSIAN_UTIL_H

#include<cmath>
#include<string>
#include<random>
#include<vector>
#include<typeindex>

#include<armadillo>

#include "PriorHessianError.h"

/**
 * 
 * 
 */
namespace prior_hessian
{

using IdxT = arma::uword;
using UVecT = arma::Col<IdxT>;
using VecT = arma::Col<double>;
using MatT = arma::Mat<double>;
using StringVecT = std::vector<std::string>; 
using TypeInfoVecT = std::vector<std::type_index>;
    
namespace constants {
    extern const double sqrt2;
    extern const double sqrt2_inv;
    extern const double sqrt2pi;
    extern const double sqrt2pi_inv;
    extern const double log2pi;
} /* namespace prior_hessian::constants */

template<class T>
T square(T t) 
{ 
    return t*t;
}


} /* namespace prior_hessian */


#endif /* PRIOR_HESSIAN_UTIL_H */
