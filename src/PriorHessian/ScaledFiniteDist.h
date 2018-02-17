/** @file ScaledFiniteDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief SemiInfiniteDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_SCALEDFINITEDIST_H
#define _PRIOR_HESSIAN_SCALEDFINITEDIST_H


#include "UnivariateDist.h"

namespace prior_hessian {


template<class Derived>
class ScaledFiniteDist : public UnivariateDist<Derived>
{
public:
   ScaledFiniteDist(std::string var_name, StringVecT&& param_desc);
   ScaledFiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc);

   double cdf(double x) const;
   double pdf(double x) const;
   double icdf(double u) const;
   double llh(double x) const;

   void set_bounds(double lbound, double ubound);
protected:
    double llh_scaling_const;
    double bounds_delta; //_ubound-_lbound
    double compute_llh_scaling_const() const;
    double convert_to_unitary_coords(double x) const;
    double convert_from_unitary_coords(double u) const;
};



template<class Derived>
ScaledFiniteDist<Derived>::ScaledFiniteDist(std::string var_name, StringVecT&& param_desc) :
    ScaledFiniteDist(0,1,var_name,std::move(param_desc))
{ }

template<class Derived>
ScaledFiniteDist<Derived>::ScaledFiniteDist(double lbound, double ubound, std::string var_name, StringVecT&& param_desc):
    UnivariateDist<Derived>(var_name, std::move(param_desc))
{ }

template<class Derived>
void ScaledFiniteDist<Derived>::set_bounds(double lbound, double ubound)
{
    if(lbound == -INFINITY) {
        std::ostringstream msg;
        msg<<"ScaledFiniteDist: lbound must be finite. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    } else if(ubound == INFINITY) {
        std::ostringstream msg;
        msg<<"ScaledFiniteDist: ubound must be finite. Got:"<<lbound;
        throw PriorHessianError("BoundsError",msg.str());
    }
    UnivariateDist<Derived>::set_bounds(lbound,ubound); //Set top-level bounds
    
    bounds_delta = ubound-lbound;
    llh_scaling_const = compute_llh_scaling_const(); //Truncation terms.
}

template<class Derived>
double ScaledFiniteDist<Derived>::compute_llh_scaling_const() const
{
    return (bounds_delta==1) ? 0 : -log(bounds_delta);
}

template<class Derived>
double ScaledFiniteDist<Derived>::llh(double x) const
{
    return static_cast<Derived const*>(this)->rllh(x) + this->llh_const + llh_scaling_const;
}

template<class Derived>
double ScaledFiniteDist<Derived>::cdf(double x) const
{
    return static_cast<Derived const*>(this)->unscaled_cdf( convert_to_unitary_coords(x) );
}

template<class Derived>
double ScaledFiniteDist<Derived>::icdf(double u) const
{
    return convert_from_unitary_coords( static_cast<Derived const*>(this)->unscaled_icdf(u) );
}

template<class Derived>
double ScaledFiniteDist<Derived>::pdf(double x) const
{
    return static_cast<Derived const*>(this)->unscaled_pdf(convert_to_unitary_coords(x)) / bounds_delta; //Correct for scaling by bounds_delta factor
}


template<class Derived>
double ScaledFiniteDist<Derived>::convert_to_unitary_coords(double x) const
{
    return (x-this->_lbound)/bounds_delta;
}

template<class Derived>
double ScaledFiniteDist<Derived>::convert_from_unitary_coords(double u) const
{
    return this->_lbound + u*bounds_delta;
}


} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_SCALEDFINITEDIST_H */
