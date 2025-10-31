#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include "fa/attention.hpp"
using namespace fa;

// 1) Invalid dropout_prob < 0
TEST(AttentionDropout, NegativeProbThrows) {
  Tensor Q = Tensor::randn({1,1,3,3}, 1);
  Tensor K = Tensor::randn({1,1,3,3}, 2);
  Tensor V = Tensor::randn({1,1,3,3}, 3);
  AttentionOpts opts; opts.dropout_prob = -0.1f;
  EXPECT_THROW(attention_forward(Q,K,V,nullptr,opts), std::invalid_argument);
}

// 2) Invalid dropout_prob > 1
TEST(AttentionDropout, GreaterThanOneThrows) {
  Tensor Q = Tensor::randn({1,1,3,3}, 4);
  Tensor K = Tensor::randn({1,1,3,3}, 5);
  Tensor V = Tensor::randn({1,1,3,3}, 6);
  AttentionOpts opts; opts.dropout_prob = 1.2f;
  EXPECT_THROW(attention_forward(Q,K,V,nullptr,opts), std::invalid_argument);
}

// 3) Valid range [0,1] → no-op (P2P)
TEST(AttentionDropout, ValidRangeNoChange) {
  Tensor Q = Tensor::randn({1,1,4,4}, 10);
  Tensor K = Q;
  Tensor V = Tensor::randn({1,1,4,4}, 11);
  AttentionOpts opts; opts.dropout_prob = 0.5f;
  Tensor O = attention_forward(Q,K,V,nullptr,opts);
  EXPECT_TRUE(std::isfinite(O.at(0,0,0,0)));
}

// 4) NaN dropout_prob should also throw
TEST(AttentionDropout, InvalidNaNProbThrows) {
  Tensor Q = Tensor::randn({1,1,3,3}, 30);
  Tensor K = Tensor::randn({1,1,3,3}, 31);
  Tensor V = Tensor::randn({1,1,3,3}, 32);

  AttentionOpts opts;
  opts.dropout_prob = std::numeric_limits<float>::quiet_NaN();  // invalid

  EXPECT_THROW(attention_forward(Q,K,V,nullptr,opts), std::invalid_argument);
}

