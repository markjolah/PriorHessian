/** @file MultivariateDist.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief MultivariateDist base class.
 */
#ifndef PRIOR_HESSIAN_MULTIVARIATEDIST_H
#define PRIOR_HESSIAN_MULTIVARIATEDIST_H

#include "PriorHessian/util.h"
#include "PriorHessian/Meta.h"
#include "PriorHessian/BaseDist.h"

namespace prior_hessian {

class MultivariateDist : public BaseDist {
public:

    MultivariateDist() {}
    
//     template<class Vec>
//     MultivariateDist(Vec &&lbound, Vec &&ubound) : 
//         _lbound(std::forward<Vec>(lbound)),
//         _ubound(std::forward<Vec>(ubound))
//     { 
//         check_bounds(lbound,ubound); 
//     }
// 
//     const NdimVecT& lbound() const { return _lbound; }
//     const NdimVecT& ubound() const { return _ubound; }

//     template<class Vec>
//     bool in_bounds(const Vec &u) const; 
//     
//     template<class Vec>
//     void set_bounds(const Vec &lbound, const Vec &ubound)
//     {
//         if(arma::any(lbound != _lbound) || arma::any(ubound != _ubound))
//             throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
//     }
//     
//     template<class Vec>
//     void set_lbound(const Vec &lbound)
//     {
//         if(arma::any(lbound != _lbound))
//             throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
//     }
// 
//     template<class Vec>
//     void set_ubound(const Vec &ubound)
//     {
//         if(arma::any(ubound != _ubound))
//             throw InvalidOperationError("MultivariateDist: Unable to set bounds.  This object is not scalable or truncatable.");
//     }

protected:
    template<class Vec>
    static void check_bounds(const Vec &lbound, const Vec &ubound)
    {
        if( !arma::all(lbound < ubound) ){ //This comparison checks for NaNs    
            std::ostringstream msg;
            msg<<"UnivariateDist::set_bounds: Invalid bounds lbound:"<<lbound.t()<<" ubound:"<<ubound.t();
            throw ParameterValueError(msg.str());
        }
    }
    
//     template<class Vec>
//     void initialize_bounds(const Vec &lbound, const Vec &ubound)
//     {
//         if(arma::any(lbound > ubound))
//             throw InvalidOperationError("MultivariateDist: initialize_bounds: got bad bounds");
//         _lbound = lbound;
//         _ubound = ubound;
//     }
//     
// private:
//     NdimVecT _lbound;
//     NdimVecT _ubound;
};

// template<class Dist>
// std::ostream& operator<<(std::ostream &out,const meta::ReturnIfSubclassOfNumericTemplateT<Dist,Dist,MultivariateDist> &dist)
// {
//     out<<"[Dist]:\n";
//     out<<"  ParamNames:[";
//     for(auto v: dist.param_names()) out<<v<<",";
//     out<<"]\n";
//     out<<"   Nparams:"<<dist.num_params()<<"\n";
//     out<<"   Params:"<<dist.params().t();
//     out<<"   Lbound:"<<dist.lbound()<<"\n";
//     out<<"   Ubound:"<<dist.ubound()<<"\n";
//     out<<"]\n";
//     return out;
// }


// template<int Ndim>
// template<class Vec>
// bool MultivariateDist<Ndim>::in_bounds(const Vec &u) const
// {
//     return arma::all(lbound()<=u) && arma::all(u<=ubound());
// }

} /* namespace prior_hessian */

#endif /* PRIOR_HESSIAN_MULTIVARIATEDIST_H */
