/** @file test_prior_hessian.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2016-2018
 * @brief Common include for all testing modules
 */
#ifndef TEST_PRIOR_HESSIAN_H
#define TEST_PRIOR_HESSIAN_H


#include "gtest/gtest.h"
#include "test_helpers/rng_environment.h"
#include <random>
#include <type_traits>
#include <armadillo>


/* Globals */
extern test_helper::RngEnvironment *env;


void check_symmetric(const arma::mat &m);
void check_positive_definite(const arma::mat &m);

inline
bool approx_equal(double a, double b, double eps)
{
    double m = std::max(fabs(a),fabs(b));
    
    return m==0 || fabs(a-b)/m<eps;
}

#endif /* TEST_PRIOR_HESSIAN_H */
