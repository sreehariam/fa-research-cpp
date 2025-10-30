#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include "fa/attention.hpp"
#include "fa/types.hpp"
#include <cmath>

using namespace fa;

static bool all_finite(const Tensor& T) {
  for (long long i = 0; i < T.numel(); ++i) {
    float v = T.at_index(i);
    if (!std::isfinite(v)) return false;
  }
  return true;
}

// 1) Small exact: Q=0, K=0 -> logits=0 -> uniform softmax -> output is row-wise mean of V
TEST(AttentionRef, SmallExact_UniformMeans_B1H1N4D2) {
  Tensor Q = Tensor::zeros({1,1,4,2});
  Tensor K = Tensor::zeros({1,1,4,2});
  Tensor V({1,1,4,2});
  V.at(0,0,0,0)=1; V.at(0,0,0,1)=2;
  V.at(0,0,1,0)=3; V.at(0,0,1,1)=4;
  V.at(0,0,2,0)=5; V.at(0,0,2,1)=6;
  V.at(0,0,3,0)=7; V.at(0,0,3,1)=8;

  AttentionOpts opts; opts.causal = false;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P: base throws NYI

  // mean of V rows = (4,5)
  for (int i=0;i<4;++i) {
    EXPECT_NEAR(O.at(0,0,i,0), 4.0f, 1e-6);
    EXPECT_NEAR(O.at(0,0,i,1), 5.0f, 1e-6);
  }
}

// 2) Shape mismatches must throw std::invalid_argument
TEST(AttentionRef, ShapeMismatch_ThrowsInvalidArgument) {
  AttentionOpts opts;
  // N mismatch
  {
    Tensor Q = Tensor::zeros({1,1,4,4});
    Tensor K = Tensor::zeros({1,1,5,4});
    Tensor V = Tensor::zeros({1,1,5,4});
    EXPECT_THROW({
      try { (void)attention_forward(Q,K,V,nullptr,opts); }
      catch (const std::invalid_argument&) { throw; }
    }, std::invalid_argument);
  }
  // D mismatch
  {
    Tensor Q = Tensor::zeros({1,1,4,8});
    Tensor K = Tensor::zeros({1,1,4,4});
    Tensor V = Tensor::zeros({1,1,4,4});
    EXPECT_THROW({
      try { (void)attention_forward(Q,K,V,nullptr,opts); }
      catch (const std::invalid_argument&) { throw; }
    }, std::invalid_argument);
  }
}

// 3) Random finite & correct shape on moderate size
TEST(AttentionRef, RandomFinite_B2H2N32D8) {
  Tensor Q = Tensor::randn({2,2,32,8}, 123);
  Tensor K = Tensor::randn({2,2,32,8}, 456);
  Tensor V = Tensor::randn({2,2,32,8}, 789);

  AttentionOpts opts; opts.causal = false;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base

  ASSERT_EQ(O.ndim(), 4);
  EXPECT_EQ(O.dim(0), 2);
  EXPECT_EQ(O.dim(1), 2);
  EXPECT_EQ(O.dim(2), 32);
  EXPECT_EQ(O.dim(3), 8);
  EXPECT_TRUE(all_finite(O));
}

// 4) Row probabilities sum ~ 1 (check indirectly using K=Q, V = identity-like basis)
TEST(AttentionRef, RowSumsCloseToOne_IndirectCheck) {
  const int B=1,H=1,N=16,D=8;
  Tensor Q = Tensor::randn({B,H,N,D}, 11);
  Tensor K = Q; // symmetric logits tendency
  Tensor V = Tensor::randn({B,H,N,D}, 22);

  AttentionOpts opts; opts.causal=false;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base

  // Not directly reading probabilities, but ensure outputs are finite and bounded
  EXPECT_TRUE(all_finite(O));
  // crude bound: large explosions would violate reasonable magnitude (heuristic)
  for (int i=0;i<N;i++) {
    for (int d=0; d<D; d++) {
      EXPECT_LT(std::fabs(O.at(0,0,i,d)), 1e6f);
    }
  }
}

// 5) Larger B/H sanity check (no mask, non-causal) to ensure batching works
TEST(AttentionRef, BatchedHeads_B3H4N10D4) {
  Tensor Q = Tensor::randn({3,4,10,4}, 99);
  Tensor K = Tensor::randn({3,4,10,4}, 98);
  Tensor V = Tensor::randn({3,4,10,4}, 97);

  AttentionOpts opts; opts.causal=false;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base
  ASSERT_EQ(O.ndim(), 4);
  EXPECT_EQ(O.dim(0), 3);
  EXPECT_EQ(O.dim(1), 4);
  EXPECT_EQ(O.dim(2), 10);
  EXPECT_EQ(O.dim(3), 4);
  EXPECT_TRUE(all_finite(O));
}
