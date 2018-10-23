/** @file SymmetricBetaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief SymmetricBetaDist class defintion
 * 
 */
#include "PriorHessian/SymmetricBetaDist.h"
#include "PriorHessian/PriorHessianError.h"

#include <sstream>
#include <cmath>
#include <limits>

#include <boost/math/special_functions/beta.hpp>

namespace prior_hessian {

const StringVecT SymmetricBetaDist::_param_names = { "beta" };
const VecT SymmetricBetaDist::_param_lbound = {0}; //Lower bound on valid parameter values 
const VecT SymmetricBetaDist::_param_ubound = {INFINITY}; //Upper bound on valid parameter values

/* Constructors */
SymmetricBetaDist::SymmetricBetaDist(double beta) 
    : UnivariateDist(0,1),
      _beta(checked_beta(beta)),
      llh_const(compute_llh_const())
{ }

/* Non-static member functions */
double SymmetricBetaDist::cdf(double x) const
{
    if(x==0) return 0;
    if(x==1) return 1;
    return boost::math::ibeta(_beta, _beta, x);
}

double SymmetricBetaDist::icdf(double u) const
{
    if(u==0) return 0;
    if(u==1) return 1;
    return boost::math::ibeta_inv(_beta, _beta, u);
}

double SymmetricBetaDist::pdf(double x) const
{
   return boost::math::ibeta_derivative(_beta, _beta, x);
}

double SymmetricBetaDist::compute_llh_const() const
{
    return -2*lgamma(_beta) - lgamma(2*_beta);//log(1/Beta(beta,beta))
}

double SymmetricBetaDist::checked_beta(double val)
{
    if(val<=0 || !std::isfinite(val)) {
        std::ostringstream msg;
        msg<<"SymmetricBetaDist: got bad beta value:"<<val;
        throw ParameterValueError(msg.str());
    }
    return val;
}
  
} /* namespace prior_hessian */
