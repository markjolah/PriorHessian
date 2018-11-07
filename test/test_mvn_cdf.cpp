/** @file test_mvn_cdf.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2017-2018
 */
#include<iostream>
#include <limits>
#include "test_prior_hessian.h"
#include "PriorHessian/mvn_cdf.h"
#include "PriorHessian/util.h"

using namespace prior_hessian;

class MVNCDFTest : public ::testing::Test {
public:    
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
    }
    static constexpr int N_samples = 41;
    static VecT sample_f;
    static VecT sample_r;
    static VecT sample_x;
    static VecT sample_y;
};


VecT MVNCDFTest::sample_f = {
    0.02260327218569867E+00,
    0.1548729518584100E+00,
    0.4687428083352184E+00,
    0.7452035868929476E+00,
    0.8318608306874188E+00,
    0.8410314261134202E+00,
    0.1377019384919464E+00,
    0.1621749501739030E+00,
    0.1827411243233119E+00,
    0.2010067421506235E+00,
    0.2177751155265290E+00,
    0.2335088436446962E+00,
    0.2485057781834286E+00,
    0.2629747825154868E+00,
    0.2770729823404738E+00,
    0.2909261168683812E+00,
    0.3046406378726738E+00,
    0.3183113449213638E+00,
    0.3320262544108028E+00,
    0.3458686754647614E+00,
    0.3599150462310668E+00,
    0.3742210899871168E+00,
    0.3887706405282320E+00,
    0.4032765198361344E+00,
    0.4162100291953678E+00,
    0.6508271498838664E+00,
    0.8318608306874188E+00,
    0.0000000000000000,
    0.1666666666539970,
    0.2500000000000000,
    0.3333333333328906,
    0.5000000000000000,
    0.7452035868929476,
    0.1548729518584100,
    0.1548729518584100,
    0.06251409470431653,
    0.7452035868929476,
    0.1548729518584100,
    0.1548729518584100,
    0.06251409470431653,
    0.6337020457912916 };
    
VecT MVNCDFTest::sample_r = {
        0.500,  0.500,  0.500,  0.500,  0.500,
        0.500, -0.900, -0.800, -0.700, -0.600,
        -0.500, -0.400, -0.300, -0.200, -0.100,
        0.000,  0.100,  0.200,  0.300,  0.400,
        0.500,  0.600,  0.700,  0.800,  0.900,
        0.673,  0.500, -1.000, -0.500,  0.000,
        0.500,  1.000,  0.500,  0.500,  0.500,
        0.500,  0.500,  0.500,  0.500,  0.500,
        0.500 };
        
VecT MVNCDFTest::sample_x = {
        -2.0, -1.0,  0.0,  1.0,  2.0,
        3.0, -0.2, -0.2, -0.2, -0.2,
        -0.2, -0.2, -0.2, -0.2, -0.2,
        -0.2, -0.2, -0.2, -0.2, -0.2,
        -0.2, -0.2, -0.2, -0.2, -0.2,
        1.0,  2.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  1.0, -1.0,
        -1.0,  1.0,  1.0, -1.0, -1.0,
        0.7071067811865475 };
        
VecT MVNCDFTest::sample_y = {
        1.0,  1.0,  1.0,  1.0,  1.0,
        1.0,  0.5,  0.5,  0.5,  0.5,
        0.5,  0.5,  0.5,  0.5,  0.5,
        0.5,  0.5,  0.5,  0.5,  0.5,
        0.5,  0.5,  0.5,  0.5,  0.5,
        0.5,  1.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  1.0, -1.0,  1.0,
        -1.0,  1.0, -1.0,  1.0, -1.0,
        0.7071067811865475 };

const double EPSILON = std::numeric_limits<double>::epsilon();
        
TEST_F(MVNCDFTest, owen_t_integral)
{
    double h,a;
    for(int n=0; n<Ntest; n++) {
        //Alternate big and small h,a values
        if(n%2==0) h = env->sample_exponential(1000);
        else  h = env->sample_exponential(1.0/1000);
        if(n%4>=2) a = env->sample_exponential(1000);
        else  a = env->sample_exponential(1.0/1000);
        double gh = unit_normal_cdf(h);
        EXPECT_GE(gh,0);
        EXPECT_LE(gh,1);
        
        //NaN values
        EXPECT_THROW(owen_t_integral(NAN,a,gh),ParameterValueError);
        EXPECT_THROW(owen_t_integral(h,NAN,gh),ParameterValueError);
        
        // Bad gh values
        EXPECT_THROW(owen_t_integral(a,h,-EPSILON),ParameterValueError);
        EXPECT_THROW(owen_t_integral(a,h,1+EPSILON),ParameterValueError);
        EXPECT_THROW(owen_t_integral(a,h,NAN),ParameterValueError);
        EXPECT_THROW(owen_t_integral(a,h,INFINITY),ParameterValueError);
        EXPECT_THROW(owen_t_integral(a,h,-INFINITY),ParameterValueError);

        // h=INFINITY
        EXPECT_EQ(owen_t_integral(INFINITY,a), 0);
        EXPECT_EQ(owen_t_integral(-INFINITY,a), 0);
        EXPECT_EQ(owen_t_integral(INFINITY,-a), 0);
        EXPECT_EQ(owen_t_integral(-INFINITY,-a), 0);

        // a=INFINITY
        EXPECT_FLOAT_EQ(owen_t_integral(h, INFINITY), .5*(1-gh));
        EXPECT_FLOAT_EQ(owen_t_integral(-h, INFINITY), .5*unit_normal_cdf(-h))<<"h: "<<h<<" gh:"<<gh;
        EXPECT_FLOAT_EQ(owen_t_integral(h, -INFINITY), -.5*(1-gh));
        EXPECT_FLOAT_EQ(owen_t_integral(-h, -INFINITY), -.5*(1-gh));

        //h=0
        EXPECT_FLOAT_EQ(owen_t_integral(0, -INFINITY), -.25);
        EXPECT_FLOAT_EQ(owen_t_integral(0, INFINITY), .25);
        EXPECT_FLOAT_EQ(owen_t_integral(0, a), atan(a)/(2*arma::datum::pi));
        EXPECT_FLOAT_EQ(owen_t_integral(0, -a), -atan(a)/(2*arma::datum::pi));
        
        //a=0
        EXPECT_EQ(owen_t_integral(h, 0), 0);
        EXPECT_EQ(owen_t_integral(-h, 0), 0);
        EXPECT_EQ(owen_t_integral(INFINITY, 0), 0);
        EXPECT_EQ(owen_t_integral(-INFINITY, 0), 0);

        //Negative Inversion formula
        EXPECT_FLOAT_EQ(owen_t_integral(-h, -a), -owen_t_integral(h, a));
        EXPECT_FLOAT_EQ(owen_t_integral(h, -a), -owen_t_integral(h, a));
        EXPECT_FLOAT_EQ(owen_t_integral(-h, a), owen_t_integral(-h, a));
        //Recipriocl Inversion formula
        double gah = unit_normal_cdf(a*h);
        double inv_val = .5*(gh+gah)-gh*gah -owen_t_integral(a*h, 1./a);
        EXPECT_LT(fabs(owen_t_integral(h, a) - inv_val), 1e-10);
        
        //a==1
        EXPECT_FLOAT_EQ(owen_t_integral(h, 1), .5*gh*(1-gh));
        EXPECT_LT(fabs(owen_t_integral(-h, 1) - .5*gh*(1-gh)), EPSILON);
    }
}

TEST_F(MVNCDFTest, owen_b_integral)
{
    double h,k,r;
    for(int n=0; n<Ntest; n++) {
        //Alternate big and small h,a values
        if(n%2==0) h = env->sample_exponential(1000);
        else  h = env->sample_exponential(1.0/1000);
        if(n%4>=2) k = env->sample_exponential(1000);
        else  k = env->sample_exponential(1.0/1000);
        r = env->sample_real(-1,1);
//         double gh = unit_normal_cdf(h);
//         double gk = unit_normal_cdf(k);
        
        //NaN values
        EXPECT_THROW(owen_b_integral(NAN,k,r),ParameterValueError);
        EXPECT_THROW(owen_b_integral(h,NAN,r),ParameterValueError);
        EXPECT_THROW(owen_b_integral(h,k,NAN),ParameterValueError);
        
        EXPECT_THROW(owen_b_integral(h,k,INFINITY),ParameterValueError);
        EXPECT_THROW(owen_b_integral(h,k,-INFINITY),ParameterValueError);
        EXPECT_THROW(owen_b_integral(h,k,-1-EPSILON),ParameterValueError);
        EXPECT_THROW(owen_b_integral(h,k,1+EPSILON),ParameterValueError);
        
        EXPECT_EQ(owen_b_integral(INFINITY,INFINITY,r), 1);
        
        EXPECT_EQ(owen_b_integral(-INFINITY,k,r), 0);
        EXPECT_EQ(owen_b_integral(-INFINITY,INFINITY,r), 0);
        EXPECT_EQ(owen_b_integral(-INFINITY,-INFINITY,r), 0);
        EXPECT_EQ(owen_b_integral(INFINITY,-INFINITY,r), 0);
        EXPECT_EQ(owen_b_integral(h,-INFINITY,r), 0);

        EXPECT_EQ(owen_b_integral(h,k,1), unit_normal_cdf(std::min(h,k)));
        EXPECT_EQ(owen_b_integral(h,k,-1), unit_normal_cdf(h) - unit_normal_cdf(-k));

        double val = owen_b_integral(h,k,r);
        EXPECT_GE(val,0)<<"h:"<<h<<" k:"<<k<<" r:"<<r<<" val:"<<val;
        EXPECT_LE(val,1)<<"h:"<<h<<" k:"<<k<<" r:"<<r<<" val:"<<val;
        
        //B(h,k,r) is symmetric in h & k
        EXPECT_FLOAT_EQ(val, owen_b_integral(k,h,r));
    }
}

TEST_F(MVNCDFTest, owen_b_fixed_values)
{
    for(int n=0; n<N_samples; n++) {
        double val = owen_b_integral(sample_x(n), sample_y(n), sample_r(n));
        EXPECT_TRUE(approx_equal(val, sample_f(n),1e-6))<<"x:"<<sample_x(n)<<" y:"<<sample_y(n)<<" r:"<<sample_r(n)<<" f:"<<sample_f(n)<<" v:"<<val;
    }
}

TEST_F(MVNCDFTest, donnelly_bvn_integral_fixed_values)
{
    for(int n=0; n<N_samples; n++) {
        double val = donnelly_bvn_integral(-sample_x(n), -sample_y(n), sample_r(n));
        double val_orig = donnelly_bvn_integral_orig(-sample_x(n), -sample_y(n), sample_r(n));
        EXPECT_TRUE(approx_equal(val, val_orig,1e-9));
        EXPECT_TRUE(approx_equal(val, sample_f(n),1e-6));
    }
}

TEST_F(MVNCDFTest, unit_normal_cdf_extrema)
{
   EXPECT_EQ(unit_normal_cdf(0), .5);
   EXPECT_EQ(unit_normal_cdf(-INFINITY), 0);
   EXPECT_EQ(unit_normal_cdf(+INFINITY), 1);
   EXPECT_EQ(unit_normal_icdf(.5), 0);
   EXPECT_EQ(unit_normal_icdf(0), -INFINITY);
   EXPECT_EQ(unit_normal_icdf(1), +INFINITY);
}

TEST_F(MVNCDFTest, donnelly_bvn_integral_extrema)
{
    for(int n=0; n<Ntest; n++) {
        double x = env->sample_exponential(1000);
        double y = env->sample_exponential(1000);
        double r = env->sample_real(-1,1);
        //All of these fail on the original version
        EXPECT_EQ(donnelly_bvn_integral(INFINITY,INFINITY,r),0);
        EXPECT_EQ(donnelly_bvn_integral(INFINITY,y,r),0);
        EXPECT_EQ(donnelly_bvn_integral(x,INFINITY,r),0);        
        EXPECT_EQ(donnelly_bvn_integral(-INFINITY,-INFINITY,r),1);
//         EXPECT_EQ(bvn_integral(-INFINITY,x,r),bvn_integral(x,-INFINITY,r));
    }
}

template<class Vec>
void print_vec(const Vec &v)
{
    int N=v.n_elem;
    std::cout<<"[";
    for(int n=0;n<N;n++) {
        std::cout<<v(n);
        if(n<N-1) std::cout<<", ";
    }
    std::cout<<"]";
}

template<class Mat>
void print_mat(const Mat &a)
{
    int N = a.n_rows;
    int M = a.n_cols;
    std::cout<<"[";
    for(int n=0;n<N;n++) for(int m=0;m<M;m++) {
        std::cout<<a(n,m);
        if(m<M-1) std::cout<<", ";
        else if(n<N-1) std::cout<<"\n";
    }
    std::cout<<"]";
}

TEST_F(MVNCDFTest, owen_bvn_cdf)
{
    for(int n=0;n<this->Ntest;n++) {
        VecT b = env->sample_normal_vec(2,0,4);
        MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,1));
        
        double v = owen_bvn_cdf(b,S);
        EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        
        VecT s = arma::sqrt(S.diag());
        VecT b0 = b/s;
        MatT S0 = S / (s*s.t());
        double v0 = owen_bvn_cdf(b0,S0);
        
        EXPECT_LE(fabs(v-v0),1E-9)<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v0:"<<v0;
    }
}

TEST_F(MVNCDFTest, donnelly_bvn_cdf)
{
    for(int n=0;n<this->Ntest;n++) {
        VecT b = env->sample_normal_vec(2,0,4);
        MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,1));
        
        double v = donnelly_bvn_cdf(b,S);
        EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        
        VecT s = arma::sqrt(S.diag());
        VecT b0 = b/s;
        MatT S0 = S / (s*s.t());
        double v0 = donnelly_bvn_cdf(b0,S0);
        
        EXPECT_LE(fabs(v-v0),1E-9)<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v0:"<<v0;
    }
}


TEST_F(MVNCDFTest, owen_vs_donnelly_bvn_cdf)
{
    for(int n=0;n<this->Ntest;n++) {
        VecT b = env->sample_normal_vec(2,0,4);
        MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,1));
        
        double v = owen_bvn_cdf(b,S);
        EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        
        VecT s = arma::sqrt(S.diag());
        VecT b0 = b/s;
        MatT S0 = S / (s*s.t());
        double v0 = owen_bvn_cdf(b0,S0);
        
        EXPECT_LE(fabs(v-v0),1E-9)<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v0:"<<v0;
        
        double d = donnelly_bvn_cdf(b,S);
        double d0 = donnelly_bvn_cdf(b0,S0);
        EXPECT_LE(fabs(v-d),1E-9)<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" d:"<<d;
        EXPECT_LE(fabs(v-d0),1E-9)<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" d0:"<<d0;    
    }
}


// TEST_F(MVNCDFTest, mc_mvn_cdf)
// {
//     for(int n=0;n<this->Ntest;n++) {
//         VecT b = env->sample_normal_vec(2,0,4);
//         MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,1));
//         MatT U = arma::chol(S);
//         double verror;
//         double v = mc_mvn_cdf(b,S,verror);
//         double v2 = owen_bvn_cdf(b,S);
//         std::cout<<"b:"<<b<<" S:"<<S<<" U:"<<U<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror<<"\n";
//         EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
//         EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
//         std::cout<<"b: "; print_vec(b);
//         std::cout<<"S: "; print_mat(S);
//         EXPECT_LE(fabs(v2-v),std::max(1e-5,2*verror))<<"b:"<<b<<" S:"<<S<<" U:"<<U<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
//     }
// }

TEST_F(MVNCDFTest, genz_fortran_2d_mvn_cdf)
{
    for(int n=0;n<this->Ntest;n++) {
        VecT b = env->sample_normal_vec(2,0,4);
        MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,1));
        MatT U = arma::chol(S);
        double verror;
        double v = genz::mvn_cdf_genz(b,S,verror);
        double v2 = owen_bvn_cdf(b,S);    
//         std::cout<<"b:"<<b<<" S:"<<S<<" U:"<<U<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
        EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
        EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
//         std::cout<<"b: "; print_vec(b);
//         std::cout<<"S: "; print_mat(S);
        EXPECT_LE(fabs(v2-v),std::max(1e-9,2*verror))<<"b:"<<b<<" S:"<<S<<" U:"<<U<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
    }
}




TEST_F(MVNCDFTest, genz_fortran_3d_mvn_cdf)
{
        MatT bs = { {0, 0, 0}, {-1,0,1}, {1,1,1}, {-1,-1,1}};
        MatT S = {{1.2, .4, -.3}, {.4, 1.9, .9}, {-.3, .9, 2.1}};
        VecT fs = {0.168399788424521, 0.0918727337962062, 0.521853874350785, 0.0559846899485406};
        int  N = bs.n_rows;
        double verror;
        for(int n=0; n<N;n++){
            VecT b = bs.row(n).t();
            double v = genz::mvn_cdf_genz(b,S,verror);
            double v2 = fs(n);
//             std::cout<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
            EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
            EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
            EXPECT_LE(fabs(v-v2),std::max(1e-9,3*verror))<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
    }
}

TEST_F(MVNCDFTest, genz_fortran_4d_mvn_cdf)
{
        MatT bs = { {0, 0, 0, 0}, {-1,0,1,-1}, {1,1,1,1}, {-1,-1,1,-1}, {0,0,0,1},  {10,10,10,10}, {3,1,1,9}};
        MatT S = {{1.2, .4, -.3, -1.}, {.4, 1.9, .9, .7}, {-.3, .9, 2.1, -.7}, {-1, .7, -.7, 3.8}} ;
        VecT fs = {0.0499313051448704, 0.00689478049922999, 0.340738068871701, 0.00649338105539796, 0.0906183246427675,0.999999984755492,0.626098152194645 };
        int  N = bs.n_rows;
        double verror;
        for(int n=0; n<N;n++){
            VecT b = bs.row(n).t();
            double v = genz::mvn_cdf_genz(b,S,verror);
            double v2 = fs(n);
//             std::cout<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
            EXPECT_GE(v,0)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
            EXPECT_LE(v,1)<<"b:"<<b<<" S:"<<S<<" v:"<<v;
            EXPECT_LE(fabs(v-v2),std::max(1e-5,3*verror))<<"b:"<<b<<" S:"<<S<<" v:"<<v<<" v2:"<<v2<<" verror:"<<verror;
    }
}




// TEST_F(MVNCDFTest, bivariate_cdf_extrema)
// {
//     for(int n=0; n<Ntest; n++) {
//         MatT S = env->sample_sigma_mat(env->sample_gamma_vec(2,1,10));
//         VecT b = {-INFINITY, -INFINITY};
//         EXPECT_EQ(bvn_cdf(b,S),0);
//     }
// }
