/** @file UnivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief UnivariateDist base class.
 */
#ifndef PRIOR_HESSIAN_UNIVARIATEDIST_H
#define PRIOR_HESSIAN_UNIVARIATEDIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {

class UnivariateDist : public BaseDist {
public:
    static constexpr IdxT num_dim() { return 1; }

    UnivariateDist(double lbound, double ubound);

    double lbound() const { return _lbound; }
    double ubound() const { return _ubound; }

    void set_bounds(double lbound, double ubound);
    void set_lbound(double lbound);
    void set_ubound(double ubound);

protected:
    static void check_bounds(double lbound, double ubound);
    /**
     * Unsafe: internally set _lbound unchecked.  For use by set_lbound functions of sub-classes only. 
     * Only used by ParetoDist. 
     */
    void set_lbound_internal(double lbound) { _lbound = lbound; } 
    
private:
    double _lbound;
    double _ubound;
    
};

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_UNIVARIATEDIST_H */
