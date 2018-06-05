#include <exception>
#include <iostream>
#include <sstream>
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
    
//     CompositeDist<RngT> dist(SymmetricBetaDist(1.01,0,8,"X"));
    CompositeDist<RngT> dist(ParetoDist(3,13,20,"X"));
    std::cout<<"Dist: "<<dist<<std::endl;
    int N=100;
        
    RngT rngT(0ULL);
    std::cout<<"Sample ["<<N<<"]\n";
    arma::vec last_s;
    for(int n=0;n<N;n++){
        auto s = dist.sample(rngT);
        std::cout<<"\n=>Sample: "<<s.t();
        if( !dist.in_bounds(s)){
            std::ostringstream msg;
            msg<<"God bad sample: "<<s.t()<<" lbound:"<<dist.lbound().t()<<" ubound:"<<dist.ubound().t();
            throw std::logic_error(msg.str());
        }
        std::cout<<"LLH: "<<dist.llh(s)<<"\n";
        std::cout<<"RLLH: "<<dist.rllh(s)<<"\n";
        std::cout<<"GRAD: "<<dist.grad(s).t();
        std::cout<<"GRAD2: "<<dist.grad2(s).t();
        std::cout<<"HESS: "<<dist.hess(s);
        
//         if(n>1){
//             std::cout<<"LLH_delta: "<<dist.llh(s)-dist.llh(last_s)<<"\n";
//             std::cout<<"RLLH_delta: "<<dist.rllh(s)-dist.rllh(last_s)<<"\n";
//         }
        last_s=s;
    }
    return 0;
}
