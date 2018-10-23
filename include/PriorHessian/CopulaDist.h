/** @file CopulaDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief CopulaDist base class.
 */
#ifndef PRIOR_HESSIAN_COPULADIST_H
#define PRIOR_HESSIAN_COPULADIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {
template<class CopulaT, class... DistTs, 
         meta::ConstructableIfIsSuperClassForAllT<UnivariateDist,DistTs...>=true, 
         meta::ConstructableIfIsSuperClass<Copula,CopulaT>=true >
class CopulaDist : public BaseDist {
public:
    static constexpr IdxT num_dim() { return sizeof...(Dists); }

    CopulaDist(double lbound, double ubound);

    VecT lbound() const { return _lbound; }
    VecT ubound() const { return _ubound; }

    void set_bounds(double lbound, double ubound);
    void set_lbound(double lbound);
    void set_ubound(double ubound);

protected:
    static void check_bounds(double lbound, double ubound);
    
private:
    VecT _lbound;
    VecT _ubound;
    
};

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_COPULADIST_H */
