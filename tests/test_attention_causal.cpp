#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include "fa/attention.hpp"
#include "fa/types.hpp"
#include <cmath>

using namespace fa;

static bool all_finite(const Tensor& T){ for(long long i=0;i<T.numel();++i) if(!std::isfinite(T.at_index(i))) return false; return true; }

// 1) Small exact: Q=K=0 -> uniform over j<=i, so output = mean of V[0..i]
TEST(AttentionCausal, SmallExact_UniformPrefixMeans_B1H1N4D2) {
  Tensor Q = Tensor::zeros({1,1,4,2});
  Tensor K = Tensor::zeros({1,1,4,2});
  Tensor V({1,1,4,2});
  V.at(0,0,0,0)=1; V.at(0,0,0,1)=2;   // row0 mean=(1,2)
  V.at(0,0,1,0)=3; V.at(0,0,1,1)=4;   // row1 mean=( (1+3)/2, (2+4)/2 )=(2,3)
  V.at(0,0,2,0)=5; V.at(0,0,2,1)=6;   // row2 mean=( (1+3+5)/3, (2+4+6)/3 )=(3,4)
  V.at(0,0,3,0)=7; V.at(0,0,3,1)=8;   // row3 mean=( (1+3+5+7)/4, (2+4+6+8)/4 )=(4,5)

  AttentionOpts opts; opts.causal = true;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base (no causal yet)

  EXPECT_NEAR(O.at(0,0,0,0), 1.0f, 1e-6); EXPECT_NEAR(O.at(0,0,0,1), 2.0f, 1e-6);
  EXPECT_NEAR(O.at(0,0,1,0), 2.0f, 1e-6); EXPECT_NEAR(O.at(0,0,1,1), 3.0f, 1e-6);
  EXPECT_NEAR(O.at(0,0,2,0), 3.0f, 1e-6); EXPECT_NEAR(O.at(0,0,2,1), 4.0f, 1e-6);
  EXPECT_NEAR(O.at(0,0,3,0), 4.0f, 1e-6); EXPECT_NEAR(O.at(0,0,3,1), 5.0f, 1e-6);
}

// 2) First token can only attend to itself when causal
TEST(AttentionCausal, FirstTokenEqualsV0_B1H1N5D3) {
  Tensor Q = Tensor::randn({1,1,5,3}, 1);
  Tensor K = Tensor::randn({1,1,5,3}, 2);
  Tensor V = Tensor::randn({1,1,5,3}, 3);

  AttentionOpts opts; opts.causal = true;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base

  for(int d=0; d<3; ++d)
    EXPECT_NEAR(O.at(0,0,0,d), V.at(0,0,0,d), 1e-5);
}

// 3) Random finite with batching/heads under causal
TEST(AttentionCausal, RandomFinite_B2H3N16D8) {
  Tensor Q = Tensor::randn({2,3,16,8}, 10);
  Tensor K = Tensor::randn({2,3,16,8}, 11);
  Tensor V = Tensor::randn({2,3,16,8}, 12);

  AttentionOpts opts; opts.causal = true;
  Tensor O = attention_forward(Q,K,V,nullptr,opts); // F2P on base
  EXPECT_TRUE(all_finite(O));
}

// 4) Causal AND padding mask together (mask keep only even j; odd masked)
TEST(AttentionCausal, CausalPlusPaddingMask) {
  const int B=1,H=2,N=8,D=4;
  Tensor Q = Tensor::randn({B,H,N,D}, 21);
  Tensor K = Tensor::randn({B,H,N,D}, 22);
  Tensor V = Tensor::randn({B,H,N,D}, 23);
  Tensor M = Tensor::zeros({B,1,1,N});
  for (int j=0;j<N;++j) if (j%2==0) M.at(0,0,0,j)=1.0f; // keep even only

  AttentionOpts opts; opts.causal = true;
  Tensor O = attention_forward(Q,K,V,&M,opts); // F2P on base
  EXPECT_TRUE(all_finite(O));
}