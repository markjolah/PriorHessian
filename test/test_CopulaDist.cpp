/** @file test_CopulaDist.cpp
 * @author Mark J. Olah (mjo\@cs.unm DOT edu)
 * @date 2018
 */
#include <cmath>
#include "gtest/gtest.h"

#include "test_univariate.h"
#include "PriorHessian/BoundsAdaptedDist.h"
#include "PriorHessian/AMHCopula.h"
#include "PriorHessian/CopulaDist.h"


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
    CopulaDistT copula_dist;
    static constexpr int Ntest = 100;
    virtual void SetUp() override {
        env->reset_rng();
        initialize_distribution_tuple(marginals);
        initialize_copula(copula);
        copula_dist.initialize_copula(copula);
        copula_dist.initialize_marginals(marginals);
    }
};

using CopulaDistTestTs = ::testing::Types<
    CopulaDist<AMHCopula, TruncatedNormalDist, TruncatedNormalDist>,
    CopulaDist<AMHCopula, TruncatedGammaDist, TruncatedGammaDist>>;
    
TYPED_TEST_CASE(CopulaDistTest, CopulaDistTestTs);

TYPED_TEST(CopulaDistTest, num_components) {
    auto &copula_dist = this->copula_dist;
    EXPECT_EQ(copula_dist.num_components(),std::tuple_size<typename TypeParam::MarginalDistTupleT>::value);
}

