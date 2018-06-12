/** @file Meta.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief Enables the use of variadic templates in interesting ways.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_META_H
#define _PRIOR_HESSIAN_META_H
#include<initializer_list>

namespace prior_hessian {
    
/** @brief Class templates to utilize sequencing behaviour of std::initializer_list expressions.
 * 
 * These class templates are intended to be used in variadic template functions to sequence the order of calls as
 * a std::initializer_list.
 * 
 */
namespace meta {
    //inline void call_in_order() { }
    template<class T>
    void call_in_order(std::initializer_list<T>) 
    { }

    template<class T>
    constexpr T sum_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{0},std::plus<T>()); }
    
    template<class T>
    constexpr T prod_in_order(std::initializer_list<T> L) 
    { return std::accumulate(L.begin(),L.end(),T{1},std::multiplies<T>()); }

    
    constexpr IdxT unordered_sum() { return 0;}
    
    template<class T>
    constexpr T unordered_sum(T i) { return i;}
    
    template<class T, class... Ts>
    constexpr auto unordered_sum(T i,Ts... args) 
    { return i + unordered_sum(args...);}
}

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_META_H */
