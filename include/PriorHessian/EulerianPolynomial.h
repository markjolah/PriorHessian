/** @file EulerianPolynomial.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief EulerianPolynomial computation .
 */

#ifndef PRIOR_HESSIAN_EULERIAN_POLYNOMIAL_H
#define PRIOR_HESSIAN_EULERIAN_POLYNOMIAL_H

#include<armadillo>


namespace prior_hessian {

    
template<long N, long M>
struct eulerian_number 
    : integral_constant<long, (N-M)*eulerian_number<N-1,M-1>{} + (M+1)*eulerian_number<N-1,M>{}> {};
template<long M> 
struct eulerian_number<0,M> : integral_constant<long,M==0> {};


namespace detail
{
    template<long N, long... I>
    VecT eulerian_polynomial()
    {
        return {eulerian_number<N,I>{},...};
    }
}

template<long N>
VecT eulerian_polynomial()
{
    return detail::eulerian_polynomial<N,std::make_integer_sequence<long,N>>();
}
    
} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_EULERIAN_POLYNOMIAL_H */
