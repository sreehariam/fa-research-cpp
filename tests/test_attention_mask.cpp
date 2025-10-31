#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include "fa/attention.hpp"
#include "fa/types.hpp"
#include <cmath>

using namespace fa;

static bool all_finite(const Tensor& T) {
  for (long long i=0;i<T.numel();++i) if (!std::isfinite(T.at_index(i))) return false;
  return true;
}

// Helper to make a (B,1,1,N) mask with 0/1 values.
static Tensor make_mask(int B, int N, const std::vector<int>& keep_idx) {
  Tensor M = Tensor::zeros({B,1,1,N});
  for (int b=0;b<B;++b)
    for (int j: keep_idx) M.at(b,0,0,j) = 1.0f;
  return M;
}

// 1) Small exact: mask keeps j=1,3; result equals mean of V rows 1 and 3 (uniform softmax on kept)
TEST(AttentionMask, SmallExact_Keep13_B1H1N4D2) {
  Tensor Q = Tensor::zeros({1,1,4,2});
  Tensor K = Tensor::zeros({1,1,4,2});
  Tensor V({1,1,4,2});
  V.at(0,0,0,0)=1; V.at(0,0,0,1)=2;   // j=0 (masked)
  V.at(0,0,1,0)=3; V.at(0,0,1,1)=4;   // j=1 (keep)
  V.at(0,0,2,0)=5; V.at(0,0,2,1)=6;   // j=2 (masked)
  V.at(0,0,3,0)=7; V.at(0,0,3,1)=8;   // j=3 (keep)

  Tensor M = make_mask(/*B=*/1, /*N=*/4, /*keep*/{1,3}); // keep 1,3

  AttentionOpts opts; opts.causal=false;
  Tensor O = attention_forward(Q,K,V,&M,opts); // F2P on base

  // mean of rows 1 and 3: ((3+7)/2,(4+8)/2)=(5,6)
  for (int i=0;i<4;++i) {
    EXPECT_NEAR(O.at(0,0,i,0), 5.0f, 1e-6);
    EXPECT_NEAR(O.at(0,0,i,1), 6.0f, 1e-6);
  }
}

// 2) Wrong mask shape -> invalid_argument
TEST(AttentionMask, BadMaskShape_Throws) {
  Tensor Q = Tensor::zeros({1,1,4,4});
  Tensor K = Tensor::zeros({1,1,4,4});
  Tensor V = Tensor::zeros({1,1,4,4});
  // Wrong N (=5)
  Tensor M = Tensor::zeros({1,1,1,5});

  AttentionOpts opts;
  EXPECT_THROW({
    try { (void)attention_forward(Q,K,V,&M,opts); }
    catch (const std::invalid_argument&) { throw; }
  }, std::invalid_argument);
}

// 3) All masked row -> exact zeros output
TEST(AttentionMask, AllMaskedRow_ReturnsZeros) {
  const int B=1,H=1,N=6,D=3;
  Tensor Q = Tensor::randn({B,H,N,D}, 21);
  Tensor K = Tensor::randn({B,H,N,D}, 22);
  Tensor V = Tensor::randn({B,H,N,D}, 23);
  Tensor M = Tensor::zeros({B,1,1,N}); // everything masked

  AttentionOpts opts; opts.causal=false;
  Tensor O = attention_forward(Q,K,V,&M,opts); // F2P on base

  for (int i=0;i<N;i++) for (int d=0;d<D;d++) {
    EXPECT_FLOAT_EQ(O.at(0,0,i,d), 0.0f);
  }
}
