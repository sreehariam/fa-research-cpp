#pragma once
#include "fa/tensor.hpp"
#include <vector>

namespace fa {

// Mask is expected to be broadcastable boolean-like (0/1) of shape (B,1,1,N).
// Base commit provides declarations only.
void apply_logits_mask_inplace(float* row_logits, int N, const Tensor& mask_row, int key_row_index);

} // namespace fa


namespace fa::mask {

// Validate mask is (B,1,1,N) matching Q/K/V (B,*,N,*)
void validate_padding_mask_b11n(const Tensor& Q, const Tensor& M);

// Apply mask to logits in-place: M[b,0,0,j] == 0 -> logits[j] = -inf
void apply_padding_mask_logits(std::vector<float>& logits,
                               const Tensor& M,
                               int b /*batch*/,
                               int N /*seq_len*/);

} // namespace fa::mask
