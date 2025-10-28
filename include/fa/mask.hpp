#pragma once
#include "fa/tensor.hpp"

namespace fa {

// Mask is expected to be broadcastable boolean-like (0/1) of shape (B,1,1,N).
// Base commit provides declarations only.
void apply_logits_mask_inplace(float* row_logits, int N, const Tensor& mask_row, int key_row_index);

} // namespace fa
