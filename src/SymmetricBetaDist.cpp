/** @file SymmetricBetaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2019
 * @brief SymmetricBetaDist class definition
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
const SymmetricBetaDist::NparamsVecT SymmetricBetaDist::_param_lbound = { 0 }; //Lower bound on valid parameter values 
const SymmetricBetaDist::NparamsVecT SymmetricBetaDist::_param_ubound = { INFINITY }; //Upper bound on valid parameter values

/* Constructors */
SymmetricBetaDist::SymmetricBetaDist(double beta) 
    : UnivariateDist(),
      _beta(checked_beta(beta)),
      llh_const_initialized(false)
{ }

/* Non-static member functions */
void  SymmetricBetaDist::set_beta(double val) 
{ 
    _beta = checked_beta(val); 
    llh_const_initialized = false;
}

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

double SymmetricBetaDist::llh(double x) const 
{ 
    if(!llh_const_initialized) initialize_llh_const();
    return rllh(x) + llh_const; 
}

void SymmetricBetaDist::initialize_llh_const() const
{
    llh_const = compute_llh_const(beta());
    llh_const_initialized = true;
}

double SymmetricBetaDist::compute_llh_const(double beta)
{
    return -2*lgamma(beta) - lgamma(2*beta);//log(1/Beta(beta,beta))
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
