/** @file PriorHessianError.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The Exception classes for the PriorHessian library
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_PRIORHESSIANERROR_H
#define _PRIOR_HESSIAN_PRIORHESSIANERROR_H

#include <exception>
#include <string>

//Enable debug assertions only when requested
#ifdef PRIOR_HESSIAN_DEBUG
    #include "PriorHessian/debug/debug_assert.h"
    #ifndef PRIOR_HESSIAN_DEBUG_LEVEL
        #define PRIOR_HESSIAN_DEBUG_LEVEL 1 // macro to control assertion level
    #endif
    namespace prior_hessian { namespace assert {
        struct handler : debug_assert::default_handler,
                        debug_assert::set_level<PRIOR_HESSIAN_DEBUG_LEVEL> 
        { };
    } } /*namespace prior_hessian::assert */
    //ASSERT_SETUP can wrap code only necessary for calling assert statements
    #ifndef ASSERT_SETUP
        #define ASSERT_SETUP(x) x
    #endif
#else
    //Expand DEBUG_ASSERT to empty
    #ifndef DEBUG_ASSERT
        #define DEBUG_ASSERT(...)
    #endif

    //Expand ASSERT_SETUP to empty
    #ifndef ASSERT_SETUP
        #define ASSERT_SETUP(...)
    #endif
#endif

namespace prior_hessian {

class PriorHessianError : public std::exception
{
protected:
    std::string _condition;
    std::string _what;
public:
    PriorHessianError(std::string condition, std::string what) 
        : _condition{std::string{"ParallelRngManager:"} + condition},
          _what{what} 
    { }
    
    const char* what() const noexcept override 
    { 
        return (_condition+" :: \""+_what+"\"").c_str(); 
    }
};
    
/** @brief Indicates a index access was out of bounds
 */
struct IndexError : public PriorHessianError 
{
    IndexError(std::string message) : PriorHessianError("IndexError",message) {}
};

struct InvalidOperationError : public PriorHessianError 
{
    InvalidOperationError(std::string message) : PriorHessianError("InvalidOperationError",message) {}
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
