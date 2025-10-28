#include "gtest/gtest.h"
#include "fa/math.hpp"
#include <vector>
#include <cmath>

namespace fa::math {
    float row_max(const float* x, int n);
    float row_sumexp_stable(const float* x, int n, float m);
}

TEST(Utils, RowMaxAndSumExp) {
    std::vector<float> v = { -2.0f, 0.0f, 3.0f, 1.0f };
    float m = fa::math::row_max(v.data(), (int)v.size());
    EXPECT_FLOAT_EQ(m, 3.0f);
    float s = fa::math::row_sumexp_stable(v.data(), (int)v.size(), m);
    // exp(-5)+exp(-3)+exp(0)+exp(-2) ~= 1 + small terms
    EXPECT_GT(s, 1.0f);
    EXPECT_LT(s, 1.2f);
}
