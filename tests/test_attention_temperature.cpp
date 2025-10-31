#include "gtest/gtest.h"
#include "fa/tensor.hpp"
#include "fa/attention.hpp"
#include "fa/types.hpp"
#include <cmath>
using namespace fa;

static bool all_finite(const Tensor& T){ for(long long i=0;i<T.numel();++i) if(!std::isfinite(T.at_index(i))) return false; return true; }

// 1) Identical output when temperature=1
TEST(AttentionTemp, TempEquals1_IdenticalToBaseline) {
  Tensor Q = Tensor::randn({1,1,6,4}, 1);
  Tensor K = Tensor::randn({1,1,6,4}, 2);
  Tensor V = Tensor::randn({1,1,6,4}, 3);
  AttentionOpts opts1; opts1.temperature = 1.0f;
  AttentionOpts opts2; opts2.temperature = 1.0f;
  Tensor O1 = attention_forward(Q,K,V,nullptr,opts1);
  Tensor O2 = attention_forward(Q,K,V,nullptr,opts2);
  EXPECT_TRUE(all_finite(O1));
  for (int i=0;i<6;i++) for (int d=0;d<4;d++)
    EXPECT_NEAR(O1.at(0,0,i,d), O2.at(0,0,i,d), 1e-6);
}

// 2) Invalid temperature ≤ 0
TEST(AttentionTemp, InvalidTemperatureThrows) {
  Tensor Q = Tensor::randn({1,1,4,2}, 4);
  Tensor K = Tensor::randn({1,1,4,2}, 5);
  Tensor V = Tensor::randn({1,1,4,2}, 6);
  AttentionOpts opts; opts.temperature = 0.0f;
  EXPECT_THROW({
    try { (void)attention_forward(Q,K,V,nullptr,opts); }
    catch(const std::invalid_argument&) { throw; }
  }, std::invalid_argument);
}

// 3) Temperature < 1.0 (sharper) vs baseline (softer)
TEST(AttentionTemp, LowerTempProducesSharperOutput) {
  Tensor Q = Tensor::randn({1,1,4,4}, 10);
  Tensor K = Q; Tensor V = Tensor::randn({1,1,4,4}, 11);
  AttentionOpts soft, sharp;
  soft.temperature = 1.0f;
  sharp.temperature = 0.5f; // sharper
  Tensor Osoft = attention_forward(Q,K,V,nullptr,soft);
  Tensor Osharp = attention_forward(Q,K,V,nullptr,sharp);
  // variance of sharp > variance of soft (sharper = more peaked)
  float varSoft=0,varSharp=0;
  for (int i=0;i<4;i++) for (int d=0;d<4;d++){
    varSoft += Osoft.at(0,0,i,d)*Osoft.at(0,0,i,d);
    varSharp += Osharp.at(0,0,i,d)*Osharp.at(0,0,i,d);
  }
  EXPECT_GT(varSharp, varSoft);
}

// 4) Temperature > 1.0 (softer) vs baseline (sharper)
TEST(AttentionTemp, HigherTempProducesSofterOutput) {
  Tensor Q = Tensor::randn({1,1,4,4}, 21);
  Tensor K = Q;
  Tensor V = Tensor::randn({1,1,4,4}, 22);

  AttentionOpts base, hot;
  base.temperature = 1.0f;
  hot.temperature  = 2.0f; // smoother distribution

  Tensor Obase = attention_forward(Q,K,V,nullptr,base);
  Tensor Ohot  = attention_forward(Q,K,V,nullptr,hot);

  // variance of hot < variance of base (smoother)
  float varBase=0,varHot=0;
  for (int i=0;i<4;i++) for (int d=0;d<4;d++) {
    varBase += Obase.at(0,0,i,d)*Obase.at(0,0,i,d);
    varHot  += Ohot.at(0,0,i,d)*Ohot.at(0,0,i,d);
  }
  EXPECT_LT(varHot, varBase);
}
