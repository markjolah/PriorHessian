/** @file PolyLog.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief Poly log computation for negative integer valued paramters.
 */

#ifndef PRIOR_HESSIAN_POLYLOG_H
#define PRIOR_HESSIAN_POLYLOG_H
namespace prior_hessian {
namespace polylog {

template<int n> 
double polylog(double z);

template<>
double polylog<1>(double z)
{
    return -std::log(1-z);
}

template<>
double polylog<0>(double z)
{
    return z/(1-z);
}

template<>
double polylog<-1>(double z)
{
    // z/(1-z)^2
    double zr = 1-z;
    return z/(zr*zr);
}

template<>
double polylog<-2>(double z)
{
    // z*(1+z)/(1-z)^3
    double zr = 1-z;
    return z*(1+z) / std::pow(zr,3);
}

template<>
double polylog<-3>(double z)
{
    // z*(1+z)/(1-z)^3
    double zr = 1-z;
    return z*(1+z) / std::pow(zr,3);
}

} /* namespace prior_hessian::polylog */
} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_POLYLOG_H */
