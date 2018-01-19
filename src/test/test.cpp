#include <iostream>
#include <random>
#include "PriorHessian/CompositeDist.h"
#include "PriorHessian/GammaDist.h"
#include "PriorHessian/ParetoDist.h"
#include "PriorHessian/NormalDist.h"
#include "PriorHessian/SymmetricBetaDist.h"
using namespace prior_hessian;

//using ParallelRngT = trng::lcg64_shift;
using RngT = std::mt19937_64;

int main()
{
    
    CompositeDist<RngT> dist(SymmetricBetaDist(3,0,8,"X"),
                             GammaDist(100, 3,"I"),
                             GammaDist(30, 3,"bg")
                             );
    std::cout<<"Dist: "<<dist<<std::endl;
    int N=3;
    
    
    RngT rngT(0ULL);
    std::cout<<"Sample ["<<N<<"]\n";
        arma::vec last_s;
        for(int n=0;n<N;n++){
            
        auto s = dist.sample(rngT);
        std::cout<<"\n=>Sample: "<<s.t();
        std::cout<<"LLH: "<<dist.llh(s)<<"\n";
        std::cout<<"RLLH: "<<dist.rllh(s)<<"\n";
        std::cout<<"LLH_COMPONENTS: "<<dist.llh_components(s).t();
        std::cout<<"RLLH_COMPONENTS: "<<dist.rllh_components(s).t();
        std::cout<<"GRAD: "<<dist.grad(s).t();
        std::cout<<"GRAD2: "<<dist.grad2(s).t();
        std::cout<<"HESS: "<<dist.hess(s);
        VecT grad(dist.num_dim(),arma::fill::zeros), grad2(dist.num_dim(),arma::fill::zeros);
        MatT hess(dist.num_dim(),dist.num_dim(),arma::fill::zeros);
        dist.grad_grad2_accumulate(s,grad,grad2);
        std::cout<<"GRAD: "<<grad.t();
        std::cout<<"GRAD2: "<<grad2.t();
        grad.zeros();
        dist.grad_hess_accumulate(s,grad,hess);
        std::cout<<"GRAD: "<<grad.t();
        std::cout<<"HESS: "<<hess;
        
        if(n>1){
            std::cout<<"LLH_delta: "<<dist.llh(s)-dist.llh(last_s)<<"\n";
            std::cout<<"RLLH_delta: "<<dist.rllh(s)-dist.rllh(last_s)<<"\n";
        }
        last_s=s;
    }
    return 0;
}
