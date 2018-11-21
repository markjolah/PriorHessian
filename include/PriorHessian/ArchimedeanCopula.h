/** @file ArchimedeanCopula.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief CopulaDist base class.
 */
#ifndef PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H
#define PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H
#include<limits>
namespace prior_hessian {

class ArchimedeanCopula {
protected:
    struct D_GenTerms {
        double log_dn_gen_t;
        double eta_n_np1_t;
    };
    struct D2_GenTerms : public D_GenTerms {
        double xi_n_t;
    };
    
    struct D_IGenTerms {
        double d1_igen_ui;
        double ieta_21_ui;
    };
    struct D2_IGenTerms : public D_IGenTerms {
        double d2_igen_ui;
        double ixi_1_ui;
    };

    struct DTheta_GenTerms {
        double log_dn_gen_t;
        double eta_0n_1n_t;
    };
    struct D2Theta_GenTerms : public DTheta_GenTerms {
        double xi_0n_t;
    };

    struct DTheta_IGenTerms {
        double sum_log_d1_igen_u = 0;
        double sum_d10_igen_u = 0;
        double sum_ieta_01_11_u = 0;
    };
    struct D2Theta_IGenTerms : public DTheta_IGenTerms {
        double sum_d20_igen_u = 0;
        double sum_ixi_01_u = 0;
    };
};
    
} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_ARCHIMEDEANCOPULA_H */
