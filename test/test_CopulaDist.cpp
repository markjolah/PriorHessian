/** @file test_CopulaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include <cmath>
#include "gtest/gtest.h"

#include "test_multivariate.h"
#include "test_univariate.h"
#include "PriorHessian/BoundsAdaptedDist.h"
#include "PriorHessian/AMHCopula.h"
#include "PriorHessian/CopulaDist.h"
#include "PriorHessian/CompositeDist.h"


using namespace prior_hessian;

template<class Copula>
void initialize_copula(Copula &c)
{
    double theta = env->sample_real(c.param_lbound(), c.param_ubound());
    c.set_theta(theta);
}

template<class CopulaDistT>
class CopulaDistTest : public ::testing::Test {
public:
    using CopulaT = typename CopulaDistT::CopulaT;
    using  MarginalDistTupleT = typename CopulaDistT:: MarginalDistTupleT;
    
    MarginalDistTupleT marginals;
    CopulaT copula;
    CopulaDistT dist;
    CompositeDist composite; //To be used for comparaison
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        initialize_distribution_tuple(marginals);
        initialize_copula(copula);
        dist.initialize_copula(copula);
        dist.initialize_marginals(marginals);
        composite.initialize(marginals);
    }
};

using CopulaDistTestTs = ::testing::Types<
    CopulaDist<AMHCopula, TruncatedNormalDist, TruncatedNormalDist>>;
//     CopulaDist<AMHCopula, TruncatedGammaDist, TruncatedGammaDist>>;
    
TYPED_TEST_CASE(CopulaDistTest, CopulaDistTestTs);

TYPED_TEST(CopulaDistTest, num_components) 
{
    auto &dist = this->dist;
    EXPECT_EQ(dist.num_components(),std::tuple_size<typename TypeParam::MarginalDistTupleT>::value);
}

TYPED_TEST(CopulaDistTest, num_dim) 
{
    auto &dist = this->dist;
    EXPECT_EQ(dist.num_dim(),std::tuple_size<typename TypeParam::MarginalDistTupleT>::value);
}

TYPED_TEST(CopulaDistTest, num_params) 
{
    auto &dist = this->dist;
    auto &composite = this->composite;
    EXPECT_EQ(dist.num_params(),1+composite.num_params());
}


// TYPED_TEST(CopulaDistTest, copy_assignment) {
//     SCOPED_TRACE("copy_assignment");
//     auto &dist = this->dist;
//     TypeParam dist_copy{};
//     dist_copy = dist;
//     check_equal(dist, dist_copy);
// }

// TYPED_TEST(CopulaDistTest, copy_construction) {
//     SCOPED_TRACE("copy_construction");
//     auto &dist = this->dist;
//     TypeParam dist_copy(dist);
//     check_equal(dist, dist_copy);
// }
