/** @file UnivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief UnivariateDist base class.
 */
#ifndef PRIOR_HESSIAN_UNIVARIATEDIST_H
#define PRIOR_HESSIAN_UNIVARIATEDIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/Meta.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {

class UnivariateDist : public BaseDist {
public:
    static constexpr IdxT num_dim() { return 1; }

    UnivariateDist(double lbound, double ubound);

    double lbound() const { return _lbound; }
    double ubound() const { return _ubound; }
    bool in_bounds(double u) const; 

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

template<class Dist>
std::ostream& operator<<(std::ostream &out,const meta::ReturnIfSubclassT<Dist,Dist,UnivariateDist> &dist)
{
    out<<"[Dist]:\n";
    out<<"  ParamNames:[";
    for(auto v: dist.param_names()) out<<v<<",";
    out<<"]\n";
    out<<"   Ndim:"<<dist.num_dim()<<"\n";
    out<<"   Nparams:"<<dist.num_params()<<"\n";
    out<<"   Params:"<<dist.params().t();
    out<<"   Lbound:"<<dist.lbound().t()<<"\n";
    out<<"   Ubound:"<<dist.ubound().t()<<"\n";
    out<<"]\n";
    return out;
}


inline
bool UnivariateDist::in_bounds(double u) const
{
    return _lbound<u && u<_ubound;
}

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_UNIVARIATEDIST_H */
