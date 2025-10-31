#include "fa/attention.hpp"
#include "fa/math.hpp"
#include "fa/mask.hpp"
#include "fa/tensor.hpp"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

namespace fa {

static void validate_core(const Tensor& Q, const Tensor& K, const Tensor& V, const AttentionOpts& opts) {
    if (opts.causal)
        throw std::invalid_argument("attention_forward: opts.causal=true not supported in PR2");
    if (Q.ndim()!=4 || K.ndim()!=4 || V.ndim()!=4)
        throw std::invalid_argument("attention_forward: Q,K,V must be 4D (B,H,N,D)");
    if (Q.dim(0)!=K.dim(0) || Q.dim(0)!=V.dim(0)) throw std::invalid_argument("B mismatch");
    if (Q.dim(1)!=K.dim(1) || Q.dim(1)!=V.dim(1)) throw std::invalid_argument("H mismatch");
    if (Q.dim(2)!=K.dim(2) || Q.dim(2)!=V.dim(2)) throw std::invalid_argument("N mismatch");
    if (Q.dim(3)!=K.dim(3) || Q.dim(3)!=V.dim(3)) throw std::invalid_argument("D mismatch");
}

Tensor attention_forward(const Tensor& Q,
                         const Tensor& K,
                         const Tensor& V,
                         const Tensor* mask,
                         const AttentionOpts& opts)
{
    validate_core(Q,K,V,opts);
    const int B=Q.dim(0), H=Q.dim(1), N=Q.dim(2), D=Q.dim(3);
    if (mask) fa::mask::validate_padding_mask_b11n(Q, *mask);

    Tensor O = Tensor::zeros({B,H,N,D});
    std::vector<float> logits(N);

    for (int b=0;b<B;++b) {
      for (int h=0; h<H; ++h) {
        for (int i=0;i<N;++i) {

          // logits[j] = Q[i]·K[j]
          for (int j=0;j<N;++j) {
            float s = 0.0f;
            for (int d=0; d<D; ++d) s += Q.at(b,h,i,d)*K.at(b,h,j,d);
            logits[j] = s;
          }

          // padding mask → -inf on masked keys
          if (mask) fa::mask::apply_padding_mask_logits(logits, *mask, b, N);

          // stable softmax
          float m = fa::math::row_max(logits.data(), N);
          // If all keys masked, m == -inf and denom will stay 0; return zeros row.
          bool all_masked = std::isinf(m) && m < 0.0f;

          float denom = all_masked ? 0.0f : fa::math::row_sumexp_stable(logits.data(), N, m);
          if (denom <= 0.0f) {
            // all masked -> leave zeros
            continue;
          }

          for (int j=0;j<N;++j) {
            float w = std::exp(std::min(80.0f, logits[j]-m)) / denom;
            if (w==0.0f) continue;
            for (int d=0; d<D; ++d)
              O.at(b,h,i,d) += w * V.at(b,h,j,d);
          }
        }
      }
    }
    return O;
}

} // namespace fa
