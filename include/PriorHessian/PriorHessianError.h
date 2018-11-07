/** @file PriorHessianError.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The Exception classes for the PriorHessian library
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_PRIORHESSIANERROR_H
#define PRIOR_HESSIAN_PRIORHESSIANERROR_H

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
    std::string condition;
    std::string what_str;
    std::string what_;
public:
    PriorHessianError(std::string condition, std::string what) 
        : condition{std::string{"PriorHessianr:"} + condition},
          what_str{condition+" :: \""+what+"\""},
          what_{what} 
    { }
    
    const char* what() const noexcept override 
    { 
        
        return what_str.c_str(); 
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

struct ParameterSizeError : public PriorHessianError 
{
    ParameterSizeError(std::string message) : PriorHessianError("ParameterSizeError",message) {}
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

struct RuntimeConvergenceError : public PriorHessianError 
{
    RuntimeConvergenceError(std::string message) : PriorHessianError("RuntimeConvergenceError",message) {}
};

struct RuntimeSamplingError : public PriorHessianError 
{
    RuntimeSamplingError(std::string message) : PriorHessianError("RuntimeSamplingError",message) {}
};


struct RuntimeTypeError : public PriorHessianError 
{
    RuntimeTypeError(std::string message) : PriorHessianError("RuntimeTypeError",message) {}
};

struct NumericalOverflowError : public PriorHessianError 
{
    NumericalOverflowError(std::string message) : PriorHessianError("NumericalOverflowError",message) {}
};

struct NotImplementedError : public PriorHessianError 
{
    NotImplementedError(std::string message) : PriorHessianError("NotImplementedError",message) {}
};


} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_PRIORHESSIANERROR_H */
