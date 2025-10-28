#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include <stdexcept>

using namespace fa;

TEST(Tensor, CreateZerosAndShape) {
    Tensor t = Tensor::zeros({1,1,4,3});
    EXPECT_EQ(t.ndim(), 4);
    EXPECT_EQ(t.dim(0), 1);
    EXPECT_EQ(t.dim(1), 1);
    EXPECT_EQ(t.dim(2), 4);
    EXPECT_EQ(t.dim(3), 3);
    EXPECT_EQ(t.numel(), 1*1*4*3);
    for (long long i = 0; i < t.numel(); ++i) {
        EXPECT_FLOAT_EQ(t.at_index(i), 0.0f);
    }
    EXPECT_TRUE(t.contiguous());
}

TEST(Tensor, BoundsCheck) {
    Tensor t = Tensor::zeros({1,1,2,2});
    EXPECT_THROW(t.at_index(-1), std::out_of_range);
    EXPECT_THROW(t.at_index(5), std::out_of_range);
}

TEST(Tensor, RandnDeterministic) {
    Tensor a = Tensor::randn({1,1,2,3}, 42);
    Tensor b = Tensor::randn({1,1,2,3}, 42);
    ASSERT_EQ(a.numel(), b.numel());
    for (long long i = 0; i < a.numel(); ++i) {
        EXPECT_FLOAT_EQ(a.at_index(i), b.at_index(i));
    }
}
