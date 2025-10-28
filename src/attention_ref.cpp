#include "fa/attention.hpp"
#include <stdexcept>

namespace fa {

Tensor attention_forward(const Tensor& Q,
                         const Tensor& K,
                         const Tensor& V,
                         const Tensor* /*mask*/,
                         const AttentionOpts& /*opts*/) {
    // Base commit: intentionally not implemented to allow F2P in PR1.
    throw std::logic_error("attention_forward: NYI in base commit");
}

} // namespace fa
