#pragma once
#include "fa/tensor.hpp"
#include "fa/types.hpp"

namespace fa {

// Forward declaration of the public API. Base commit throws NYI.
Tensor attention_forward(const Tensor& Q,
                         const Tensor& K,
                         const Tensor& V,
                         const Tensor* mask,
                         const AttentionOpts& opts);

} // namespace fa
