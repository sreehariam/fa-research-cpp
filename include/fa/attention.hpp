#pragma once
#include "fa/types.hpp"
#include "fa/tensor.hpp"

namespace fa {

// Forward declaration of the public API. Base commit throws NYI.
Tensor attention_forward(const Tensor& Q,
                         const Tensor& K,
                         const Tensor& V,
                         const Tensor* mask,
                         const AttentionOpts& opts);

} // namespace fa
