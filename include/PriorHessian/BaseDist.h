/** @file BaseDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017
 * @brief The Base classes for UnivariateDist and MultivariateDist.
 * 
 * Distinguishing univariate from multivariate distributions is important as the types
 * of most functions are different (scalar vs vector).  For maximum efficiency we wish for
 * scalar types to not be forced to used size=1 vectors, but instead call directly with double
 * arguments.
 * 
 * 
 */
#ifndef _PRIOR_HESSIAN_BASEDIST_H
#define _PRIOR_HESSIAN_BASEDIST_H


#include<armadillo>

#include "PriorHessian/util.h"
#include "PriorHessian/PriorHessianError.h"

namespace prior_hessian {
 
//Forward decl
template<class RngT>
class CompositeDist;

/** @brief Encapsulates the common functionality of UnivariateDist and MultivariateDist
 */
class BaseDist
{
 template<class RngT> friend class CompositeDist;
public:
    BaseDist(StringVecT &&param_names);
    const StringVecT& param_names() const;
    IdxT num_params() const;    
    void set_param_names(const StringVecT& desc); 
protected:
    template<class IterT> void append_param_names(IterT& p) const;
    template<class IterT> void set_param_names(IterT& d); 
    StringVecT _param_names;
};


inline
BaseDist::BaseDist(StringVecT &&param_names) : 
    _param_names(std::move(param_names)) 
{ }

inline
const StringVecT& BaseDist::param_names() const 
{ return _param_names; }

inline
IdxT BaseDist::num_params() const 
{ return _param_names.size(); }

template<class IterT>
void BaseDist::append_param_names(IterT& p) const 
{ 
    p = std::copy(_param_names.cbegin(),_param_names.cend(),p); //Make sure to update p to new position
} 

inline
void BaseDist::set_param_names(const StringVecT& desc) 
{ 
    auto iter = desc.cbegin();
    set_param_names(iter);
}

template<class IterT>
void BaseDist::set_param_names(IterT& d) 
{ for(IdxT i=0; i<num_params(); i++) _param_names[i] = *d++; }



} /* namespace prior_hessian */
#endif /* _PRIOR_HESSIAN_BASEDIST_H */
