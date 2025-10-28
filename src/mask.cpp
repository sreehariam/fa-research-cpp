#include "fa/mask.hpp"
#include <stdexcept>

namespace fa {

void apply_logits_mask_inplace(float* /*row_logits*/, int /*N*/, const Tensor& /*mask_row*/, int /*key_row_index*/) {
    // Base commit: no-op. PR2 will implement masking logic.
}

} // namespace fa
