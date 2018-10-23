/** @file MultivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief MultivariateDist base class.
 */
#ifndef PRIOR_HESSIAN_MULTIVARIATEDIST_H
#define PRIOR_HESSIAN_MULTIVARIATEDIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {

class MultivariateDist : public BaseDist {
public:
    MultivariateDist(int Ndim) : 
        _lbound(Ndim),
        _ubound(Ndim)
    { 
        _lbound.fill(-INFINITY);
        _ubound.fill(INFINITY);
    }
    
    template<class Vec>
    MultivariateDist(Vec lbound, Vec ubound) : 
        _lbound(lbound),
        _ubound(ubound)
    { 
        check_bounds(lbound,ubound); 
    }

    VecT lbound() const { return _lbound; }
    VecT ubound() const { return _ubound; }

    template<class Vec>
    void set_bounds(const Vec &lbound, const Vec &ubound)
    {
        if(lbound != _lbound || ubound != _ubound)
            throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
    }
    
    template<class Vec>
    void set_lbound(const Vec &lbound)
    {
        if(lbound != _lbound)
            throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
    }

    template<class Vec>
    void set_ubound(const Vec &ubound)
    {
        if(ubound != _ubound)
            throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
    }

protected:
    template<class Vec>
    static void check_bounds(const Vec &lbound, const Vec &ubound)
    {
    if( !(lbound < ubound) ){ //This comparison checks for NaNs    
        std::ostringstream msg;
        msg<<"UnivariateDist::set_bounds: Invalid bounds lbound:"<<lbound<<" ubound:"<<ubound;
        throw ParameterValueError(msg.str());
    }
}
    
private:
    VecT _lbound;
    VecT _ubound;
};

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MULTIVARIATEDIST_H */
