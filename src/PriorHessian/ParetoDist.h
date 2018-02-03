/** @file ParetoDist.h
 * @author Mark J. Olah (mjo\@cs.unm.edu)
 * @date 2017
 * @brief ParetoDist class declaration and templated methods
 * 
 */
#ifndef _PRIOR_HESSIAN_PARETODIST_H
#define _PRIOR_HESSIAN_PARETODIST_H

#include "TruncatingDist.h"

namespace prior_hessian {

/** @brief Pareto dist with infinite upper bound.
 * 
 */
class ParetoDist : public PositiveSemiInfiniteDist<ParetoDist>
{

public:
    ParetoDist(double alpha, double lbound, std::string var_name);
    ParetoDist(double alpha, double lbound, std::string var_name, StringVecT&& param_desc);
    ParetoDist(double alpha, double lbound, double ubound, std::string var_name);
    ParetoDist(double alpha, double lbound, double ubound, std::string var_name, StringVecT&& param_desc);
     constexpr static IdxT num_params();
    static StringVecT make_default_param_desc(std::string var_name);
    
    template<class IterT> void append_params(IterT& p) const;
    template<class IterT> void set_params(IterT& p);   
        
    double rllh(double x) const;
    double grad(double x) const;
    double grad2(double x) const;
    void grad_grad2_accumulate(double x, double &g, double &g2) const;

    double compute_llh_const() const;
    double unbounded_cdf(double x) const;
    double unbounded_icdf(double u) const;
    double unbounded_pdf(double x) const;

protected:
    double alpha; //distribution shape
    
    static void check_params(double alpha_val); 
};

inline
ParetoDist::ParetoDist(double alpha, double lbound, std::string var_name) :
    ParetoDist(alpha,lbound,INFINITY,var_name,make_default_param_desc(var_name))
{ }

inline
ParetoDist::ParetoDist(double alpha, double lbound, std::string var_name, StringVecT&& param_desc) :
    ParetoDist(alpha,lbound,INFINITY,var_name,std::move(param_desc))
{ }

inline
ParetoDist::ParetoDist(double alpha, double lbound, double ubound, std::string var_name) :
    ParetoDist(alpha,lbound,ubound,var_name,make_default_param_desc(var_name))
{ }

inline
ParetoDist::ParetoDist(double alpha, double lbound, double ubound, std::string var_name, StringVecT&& param_desc) :
        PositiveSemiInfiniteDist<ParetoDist>(lbound, ubound, var_name,std::move(param_desc)),
        alpha(alpha)
{
    this->set_bounds(lbound,ubound);
    this->llh_const = compute_llh_const();
}

constexpr
IdxT ParetoDist::num_params()
{ 
    return 1; 
}

inline
StringVecT ParetoDist::make_default_param_desc(std::string var_name)
{
    return {std::string("alpha_") + var_name};
}

inline
double ParetoDist::unbounded_cdf(double x) const
{
    return 1-pow(get_lbound()/x,alpha);
}

inline
double ParetoDist::unbounded_icdf(double u) const
{
    return get_lbound() / pow(1-u,1/alpha);
}

inline
double ParetoDist::unbounded_pdf(double x) const
{
    return alpha/x * pow(get_lbound()/x,alpha);
}

inline
double ParetoDist::compute_llh_const() const
{
    return log(alpha) + alpha*log(get_lbound());
}

inline
double ParetoDist::rllh(double x) const
{
    return -(alpha+1)*log(x);
}

inline
double ParetoDist::grad(double x) const
{
    return -(alpha+1)/x;
}

inline
double ParetoDist::grad2(double x) const
{
    return (alpha+1)/(x*x);
}

inline
void ParetoDist::grad_grad2_accumulate(double x, double &g, double &g2) const
{
    double ap1ox = (alpha+1)/x;
    g  -= ap1ox ;   // -(alpha+1)/x
    g2 += ap1ox/x;  // (alpha+1)/x^2
}

template<class IterT>
void ParetoDist::append_params(IterT& p) const 
{ 
    *p++ = alpha;
} 

template<class IterT>
void ParetoDist::set_params(IterT& p) 
{ 
    double alpha_val = *p++;
    check_params(alpha_val);
    alpha = alpha_val;
    llh_const = compute_llh_const();
}     

inline
void ParetoDist::check_params(double alpha_val) 
{ 
    if(alpha_val<=0 || !std::isfinite(alpha_val)) {
        std::ostringstream msg;
        msg<<"ParetoDist::set_params: got bad alpha value:"<<alpha_val;
        throw PriorHessianError("BadParameter",msg.str());
    }
}
    
} /* namespace prior_hessian */

#endif /* _PRIOR_HESSIAN_PARETODIST_H */
