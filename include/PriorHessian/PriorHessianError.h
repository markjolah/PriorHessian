/** @file PriorHessianError.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The Exception classes for the PriorHessian library
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_PRIORHESSIANERROR_H
#define _PRIOR_HESSIAN_PRIORHESSIANERROR_H

#include<string>

#include<BacktraceException/BacktraceException.h>

namespace prior_hessian {
 
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
struct ParameterNameUniquenessError : public PriorHessianError 
{
    ParameterNameUniquenessError(std::string message) : PriorHessianError("ParameterNameError",message) {}
};

struct RuntimeTypeError : public PriorHessianError 
{
    RuntimeTypeError(std::string message) : PriorHessianError("RuntimeTypeError",message) {}
};

struct NumericalOverflowError : public PriorHessianError 
{
    NumericalOverflowError(std::string message) : PriorHessianError("NumericalOverflowError",message) {}
};

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_PRIORHESSIANERROR_H */
