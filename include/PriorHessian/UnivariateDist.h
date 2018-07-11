/** @file UnivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief UnivariateDist base class.
 */
#ifndef _PRIOR_HESSIAN_UNIVARIATEDIST_H
#define _PRIOR_HESSIAN_UNIVARIATEDIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {

class UnivariateDist : public BaseDist {
public:
    static constexpr IdxT num_dim();

    double lbound() const; 
    double ubound() const;

    void set_bounds(double lbound, double ubound);
    void set_lbound(double lbound);
    void set_ubound(double ubound);
    
protected:
    double _lbound;
    double _ubound;

    UnivariateDist(double lbound, double ubound);
    
    static void check_bounds(double lbound, double ubound);
};

constexpr 
IdxT UnivariateDist::num_dim()
{ return 1; }

inline
double UnivariateDist::lbound() const
{ return _lbound; }

inline
double UnivariateDist::ubound() const
{ return _ubound; }

} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_UNIVARIATEDIST_H */
