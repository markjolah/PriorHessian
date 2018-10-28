/** @file debug_bvn.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 * @brief NormalDist class defintion
 * 
 */
#include "PriorHessian/mvn_cdf.h"

#include <cmath>
#include <limits>

#include <armadillo>

#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>

using namespace prior_hessian;

void test_bvn()
{
    const int Nsample=100;
    double ak = 0;
    double ah = 0;
    double r=0.2;
    for(int n=0; n<Nsample; n++) {
        ak += 0.1;
        ah += 0.1;
        double b = bvn_integral(ak,ah,r);
        std::cout<<"ak:"<<ak<<" ah:"<<ah<<" r:"<<r<<" b:"<<b<<"\n";
    }
}

int main()
{
    test_bvn();
    
}
