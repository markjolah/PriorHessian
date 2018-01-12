#include <iostream>
#include <random>
#include "CompositeDist.h"
#include "SymmetricBetaDist.h"
#include "GammaDist.h"
using namespace prior_hessian;

//using ParallelRngT = trng::lcg64_shift;
using RngT = std::mt19937_64;

int main()
{
    
    CompositeDist<RngT> dist(SymmetricBetaDist(3,0,8,"x"),
                                     GammaDist(100, 3,"I"),
                                     GammaDist(3, 3,"bg"));
    std::cout<<"Dist: "<<dist<<std::endl;
    int N=10;
    std::cout<<" pdf: "<<dist.pdf(VecT{1,10,10})<<std::endl;
    
//     CompositeDist<RngT> dist2=dist;
    
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
        std::cout<<"Hess: "<<hess;
        
        if(n>1){
            std::cout<<"LLH_delta: "<<dist.llh(s)-dist.llh(last_s)<<"\n";
            std::cout<<"RLLH_delta: "<<dist.rllh(s)-dist.rllh(last_s)<<"\n";
        }
        last_s=s;
    }
    return 0;
}
