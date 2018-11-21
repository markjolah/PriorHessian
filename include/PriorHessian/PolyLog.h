/** @file PolyLog.h
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017 - 2018
 * @brief Poly log computation for negative integer valued paramters.
 */

#ifndef PRIOR_HESSIAN_POLYLOG_H
#define PRIOR_HESSIAN_POLYLOG_H
namespace prior_hessian {
namespace polylog {

    
template<int n> 
double euler_polynomial(double z);

template<>
double euler_polynomial<0>(double z)
{
    return 1;
}

template<>
double euler_polynomial<1>(double z)
{
    return 1+z;
}

template<>
double euler_polynomial<2>(double z)
{
    return 1+z*(4+z);
}

template<>
double euler_polynomial<3>(double z)
{
    return 1+z*(11+z*(11+z));
}

template<>
double euler_polynomial<4>(double z)
{
    return 1+z*(26+z*(66+z*(26+z)));
}

template<>
double euler_polynomial<5>(double z)
{
    return 1+z*(57+z*(302+z*(302+z*(57+z))));
}

template<>
double euler_polynomial<6>(double z)
{
    return 1+z*(120+z*(1191+z*(2416+z*(1191+z*(120+z)))));
}

template<>
double euler_polynomial<7>(double z)
{
    return 1+z*(247+z*(4293+z*(15619+z*(15619+z*(4293+z*(247+z))))));
}

template<>
double euler_polynomial<8>(double z)
{
    return 1+z*(502+z*(14608+z*(88234+z*(156190+z*(88234+z*(14608+z*(502+z)))))));
}
template<>
double euler_polynomial<9>(double z)
{
    return 1+z*(1013+z*(47840+z*(455192+z*(1310354+z*(1310354+z*(455192+z*(47840+z*(1013+z))))))));
}
    
    
template<int n>
double poly_A171692(double z);

template<>
double poly_A171692<0>(double z)
{
    return 1;
}

template<>
double poly_A171692<2>(double z)
{
    return 1+z*(10+z);
}

template<>
double poly_A171692<4>(double z)
{
    return 1+z*(56+z*(246+z*(56+z)));
}

template<>
double poly_A171692<6>(double z)
{
    return 1+z*(246+z*(4047+z*(11572+z*(4047+z*(246+z)))));
}

template<>
double poly_A171692<8>(double z)
{
    return 1+z*(1012+z*(46828+z*(408364+z*(901990+z*(408364+z*(46828+z*(1012+z)))))));
}

template<>
double poly_A171692<10>(double z)
{
    return 1+z*(4082+z*(474189+z*(9713496+z*(56604978+z*(105907308+z*(56604978+z*(9713496+z*(474189+z*(4082+z)))))))));
}

    
template<int n> 
double polylog(double z);

template<>
double polylog<1>(double z)
{
    return -std::log(1-z);
}

template<>
double polylog<0>(double z)
{
    return z/(1-z);
}

template<>
double polylog<-1>(double z)
{
    // z/(1-z)^2
    double zr = 1-z;
    return z/(zr*zr);
}

template<>
double polylog<-2>(double z)
{
    // z*(1+z)/(1-z)^3
    double zr = 1-z;
    return z*(1+z) / std::pow(zr,3);
}

template<>
double polylog<-3>(double z)
{
     // (z+4z^2+z^3) / (1-z)^4
    double zr = 1-z;
    return euler_polynomial<2>(z) / std::pow(zr,4);
}

template<>
double polylog<-4>(double z)
{
    // (z+11*z^2+11*z^3+z^4) / (1-z)^5
    double zr = 1-z;
    return euler_polynomial<3>(z) / std::pow(zr,5);
}

template<>
double polylog<-5>(double z)
{
    // (z+26*z^2+66*z^3+26*z^4+z^5) / (1-z)^6
    double zr = 1-z;
    return euler_polynomial<4>(z) / std::pow(zr,6);
}

template<>
double polylog<-6>(double z)
{
    // (z+57*z^2+302*z^3+302*z^4+57*z^5+z^6) / (1-z)^7
    double zr = 1-z;
    return euler_polynomial<5>(z) / std::pow(zr,7);
}

template<>
double polylog<-7>(double z)
{
    // (z+120*z^2+1191*z^3+2416*z^4+1191*z^5+120*z^6+z^7) / (1-z)^8
    double zr = 1-z;
    return euler_polynomial<6>(z) / std::pow(zr,8);
}

template<>
double polylog<-8>(double z)
{
    // (z+247*z^2+4293*z^3+15619*z^4+15619*z^5+4293*z^6+247*z^7+z^8) / (1-z)^9
    double zr = 1-z;
    return euler_polynomial<7>(z) / std::pow(zr,9);
}

template<>
double polylog<-9>(double z)
{
    // (z+502*z^2+14608*z^3+88234*z^4+156190*z^5+88234*z^6+14608*z^7+502*z^8+z^9) / (1-z)^10
    double zr = 1-z;
    return euler_polynomial<8>(z) / std::pow(zr,10);
}

template<>
double polylog<-10>(double z)
{
    // (z+1013*z^2+47840*z^3+455192*z^4+1310345*z^5+1310345*z^6+455192*z^7+47840*z^8+1013*z^9+z^10) / (1-z)^11
    double zr = 1-z;
    return euler_polynomial<9>(z) / std::pow(zr,11);
}



template<int numer, int denom> 
double polylog_ratio(double z);


template<>
double polylog_ratio<-2,-1>(double z)
{
    return (1+z)/(1-z);
}

template<>
double polylog_ratio<-3,-2>(double z)
{
    return euler_polynomial<2>(z) / (1-z*z);
}

template<>
double polylog_ratio<-4,-3>(double z)
{
    return (1+z)*poly_A171692<2>(z) / ((1-z)*euler_polynomial<2>(z));
}

template<>
double polylog_ratio<-5,-4>(double z)
{
    return -1 - 5/(z-1) - 1/(1+z)-(2+10*z)/poly_A171692<2>(z);
}

template<>
double polylog_ratio<-6,-5>(double z)
{
    return (1+z)*poly_A171692<4>(z) / ((1-z)*euler_polynomial<4>(z));
}

template<>
double polylog_ratio<-7,-6>(double z)
{
    return -1 - 7/(z-1) - 1/(1+z) - 4*(1+z*(24+z*(123+z*14))) / poly_A171692<4>(z);
}

template<>
double polylog_ratio<-8,-7>(double z)
{
    return (1+z)*poly_A171692<6>(z) / ((1-z)*euler_polynomial<6>(z))
}

template<>
double polylog_ratio<-9,-8>(double z)
{
    return -1 - 9/(z-1) - 1/(1+z) - (6*(1+z*(205+z*(2698+z*(5786+z*(1349+41*z)))))) / 
                                    poly_A171692<6>(z);
}

template<>
double polylog_ratio<-10,-9>(double z)
{
    return  (1+z)*poly_A171692<8>(z) / ((1-z)*euler_polynomial<8>(z));
}


template<>
double polylog_ratio<-11,-10>(double z)
{
    return -1 - 11/(z-1) - 1/(1+z) - 
            (4*(2+z*(1771+z*(70242+z*(510455+z*(901990+z*(306273+z*(23414+253*z)))))))) / 
            poly_A171692<8>(z);
}


/*
 * Li(-n-2; z) / Li(-n; z) 
 * 
 */

template<int n>
double polylog_second_derivative_ratio(double z);

template<>
double polylog_second_derivative_ratio<2>(double z)
{
    return 4*t*(1+t*(1+t)) / square(t*t-1);
}

template<>
double polylog_second_derivative_ratio<3>(double z)
{
    return 4*t*(2+t*(7+t*(18+t*(7+2*t)))) / square((t-1)*(1+t*(4+t)));
}

template<>
double polylog_second_derivative_ratio<4>(double z)
{
    return 4*t*(4+t*(33+t*(192+t*(262+t*(192+t*(33+4*t)))))) /
           square((t-1)*(1+t)*poly_A171692<2>(z));
}

template<>
double polylog_second_derivative_ratio<5>(double z)
{
    return 4*t*(8+t*(131+t*(1574+t*(4997+t*(8180+t*(4997+t*(1574+t*(131+8*t)))))))) / 
            square((t-1)*euler_polynomial<4>(z));
}

template<>
double polylog_second_derivative_ratio<6>(double z)
{
    return 4*t*(16+t*(473+t*(11136+t*(69084+t*(220272+t*(305238+t*(220272+t*(69084+t*(11136+t*(473+16*t)))))))))) / 
            square((t-1)*(1+t)*poly_A171692<4>(z));
}

template<>
double polylog_second_derivative_ratio<7>(double z)
{
    return 4*t*(32+t*(1611+t*(72378+t*(794887+t*(4494888+t*(11769390+t*(16536828+t*(11769390+t*(4494888+t*(794887+t*(72378+t*(1611+32*t)))))))))))) /
            square((t-1)*euler_polynomial<6>(z));
}

template<>
double polylog_second_derivative_ratio<8>(double z)
{
    return 4*t*(64+t*(5281+t*(448300+t*(8166146+t*(76827644+t*(343391567+t*(841117848+t*(1117916700+t*(841117848+t*(343391567+t*(76827644+t*(8166146+t*(448300+t*(5281+64*t)))))))))))))) /
            square((t-1)*(1+t)*poly_A171692<6>(z));
}

template<>
double polylog_second_derivative_ratio<9>(double z)
{
    return 4*t*(128+t*(16867+t*(2705790+t*(77976425+t*(1167126460+t*(8342539251+t*(33065326754+t*(73705170625+t*(96483011400+t*(73705170625+t*(33065326754+t*(8342539251+t*(1167126460+t*(77976425+t*(2705790+t*(16867+128*t)))))))))))))))) /
            square((t-1)*euler_polynomial<8>(z));
}

template<>
double polylog_second_derivative_ratio<10>(double z)
{
    return 4*t*(256+t*(52905+t*(16129320+t*(709163544+t*(16356487800+t*(178667479932+t*(1087364443560+t*(3793777308840+t*(7957216227768+t*(10144306372150+t*(7957216227768+t*(3793777308840+t*(1087364443560+t*(178667479932+t*(16356487800+t*(709163544+t*(16129320+t*(52905+256*t)))))))))))))))))) /
            square((t-1)*(1+t)*poly_A171692<8>(z));
}

} /* namespace prior_hessian::polylog */
} /* namespace prior_hessian */
#endif /* PRIOR_HESSIAN_POLYLOG_H */
