/** @file Meta.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief Enables the use of variadic templates in interesting ways.
 * 
 * 
 */
#ifndef PRIOR_HESSIAN_META_H
#define PRIOR_HESSIAN_META_H

#include <functional>
#include <initializer_list>
#include <cstdint>

namespace prior_hessian {
    
/** @brief Class templates to utilize sequencing behaviour of std::initializer_list expressions.
 * 
 * These class templates are intended to be used in variadic template functions to sequence the order of calls as
 * a std::initializer_list.
 * 
 */
namespace meta {

    /** NOOP function which is used to ensure call order on a variadic sequence of function calls 
     *
     */
    template<class T>
    void call_in_order(std::initializer_list<T>) 
    { }

    template <class InputIterator, class ResultT, class BinaryOperation>
    constexpr ResultT constexpr_accumulate (InputIterator first, InputIterator last, ResultT init, BinaryOperation op)
    {
        for(; first!=last; ++first)  init = op(init, *first);
        return init;
    }

    constexpr inline bool logical_and_in_order(std::initializer_list<bool> L) 
    { return constexpr_accumulate(L.begin(),L.end(),true,std::logical_and<bool>()); }
    
    template<class T>
    constexpr T sum_in_order(std::initializer_list<T> L) 
    { return constexpr_accumulate(L.begin(),L.end(),T{0},std::plus<T>()); }
    
    template<class T>
    constexpr T prod_in_order(std::initializer_list<T> L) 
    { return constexpr_accumulate(L.begin(),L.end(),T{1},std::multiplies<T>()); }
    
    template<class...> struct conjunction : std::true_type { };
    template<class B1> struct conjunction<B1> : B1 { };
    template<class B1, class... Bn>
    struct conjunction<B1, Bn...> 
        : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};
    
    template<class...> struct disjunction : std::true_type { };
    template<class B1> struct disjunction<B1> : B1 { };
    template<class B1, class... Bn>
    struct disjunction<B1, Bn...> 
        : std::conditional_t<bool(B1::value), disjunction<Bn...>, B1> {};

    template<template <typename...> class, typename>
    struct is_template_of : std::false_type { };

    template<template <typename...> class ClassTemplate, typename... Ts>
    struct is_template_of<ClassTemplate, ClassTemplate<Ts...>> : std::true_type { };

    /* overload for integer templates */
    template<template <int...> class, typename>
    struct is_numeric_template_of : std::false_type { };

    template<template <int...> class ClassNumericTemplate, int... Is>
    struct is_numeric_template_of<ClassNumericTemplate, ClassNumericTemplate<Is...>> : std::true_type { };

    
    template<class ReturnT, class BoolT> 
    using ReturnIfT = std::enable_if_t<BoolT::value,ReturnT>;

    template<bool val>
    using ConstructableIf = std::enable_if_t<val,bool>; /* Uses a non-type template paramter for SFINAE */
    
    template<class T,class BaseT> 
    using EnableIfSubclassT = std::enable_if_t<
        std::is_base_of<std::remove_reference_t<BaseT>,std::remove_reference_t<T>>::value >;

    template<class T,class SelfT> 
    using EnableIfNotIsSelfT = std::enable_if_t< !std::is_same<std::decay_t<T>,SelfT>::value >;

        
    template<class ReturnT, class T,class BaseT> 
    using ReturnIfSubclassT = std::enable_if_t<
        std::is_base_of<std::remove_reference_t<BaseT>,std::remove_reference_t<T>>::value, ReturnT>;

    template<class BaseT, class... Ts> 
    using EnableIfIsSuperclassOfAllT = std::enable_if_t<conjunction< 
        std::is_base_of<std::remove_reference_t<BaseT>,std::remove_reference_t<Ts>> ... >::value >;

    template<class T, template <typename...> class ClassTemplate> 
    using EnableIfInstantiatedFromT = std::enable_if_t<
                is_template_of<ClassTemplate, std::remove_reference_t<T>>::value >;
    
    template<class T, template <int> class ClassTemplate> 
    using EnableIfInstantiatedFromNumericT = std::enable_if_t<
                is_numeric_template_of<ClassTemplate, std::remove_reference_t<T>>::value, std::remove_reference_t<T> >;
                
    template<class T, template <typename...> class ClassTemplate> 
    using EnableIfNotInstantiatedFromT = std::enable_if_t<
                !is_template_of<ClassTemplate, std::remove_reference_t<T>>::value >;
                
    template<class ReturnT, class TestT, template <typename...> class ClassTemplate> 
    using ReturnIfInstantiatedFromT = std::enable_if_t< 
        is_template_of<ClassTemplate, std::remove_reference_t<TestT>>::value, ReturnT>;

    template<class ReturnT, class TestT, template <typename...> class ClassTemplate> 
    using ReturnIfNotInstantiatedFromT = std::enable_if_t< 
        !is_template_of<ClassTemplate, std::remove_reference_t<TestT>>::value, ReturnT>;

    template<template <typename> class ClassTemplate, class... Ts> 
    using EnableIfIsTemplateForAllT = std::enable_if_t< conjunction< 
        is_template_of<ClassTemplate,std::remove_reference_t<Ts>> ... >::value >;

    template<template <typename> class ClassTemplate, class... Ts> 
    using ConstructableIfIsTemplateForAllT = std::enable_if_t< conjunction< 
        is_template_of<ClassTemplate,std::remove_reference_t<Ts>> ... >::value, bool>;

    template<class SuperClass, class T> 
    using ConstructableIfIsSuperClassT = std::enable_if_t< 
        std::is_base_of<std::remove_reference_t<SuperClass>,std::remove_reference_t<T>>::value, bool>;
    
    template<class SuperClass, class... Ts> 
    using ConstructableIfIsSuperClassForAllT = std::enable_if_t< conjunction< 
        std::is_base_of<std::remove_reference_t<SuperClass>,std::remove_reference_t<Ts>>... >::value, bool>;
        
    template<class T> 
    using EnableIfIsNotTupleT = std::enable_if_t< !is_template_of<std::tuple,std::remove_reference_t<T>>::value >;

    template<class... Ts>
    using EnableIfNonEmpty = std::enable_if_t< (sizeof...(Ts)>0) >;
    
    template<class... Ts> 
    using EnableIfAllAreNotTupleT = std::enable_if_t< !disjunction< 
        is_template_of<std::tuple,std::remove_reference_t<Ts>>... >::value >;

    template<class SelfT, class T> 
    using EnableIfIsNotTupleAndIsNotSelfT = 
        std::enable_if_t< !is_template_of<std::tuple,std::remove_reference_t<T>>::value
                          && !std::is_same<std::decay_t<T>,SelfT>::value >;

    template<class SelfT, class... Ts> 
    using ConstructableIfAllAreNotTupleAndAreNotSelfT = 
        std::enable_if_t< !disjunction<is_template_of<std::tuple,std::remove_reference_t<Ts>>...>::value
                          && !disjunction<std::is_same<std::decay_t<Ts>,SelfT>...>::value, bool>;
    
    template<class Dist, class BaseDist> 
    using DerivedFrom = std::enable_if_t<std::is_base_of<std::decay_t<BaseDist>,std::decay_t<Dist>>::value,std::decay_t<Dist>>;

}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_META_H */
