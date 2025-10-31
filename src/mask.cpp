#include "fa/mask.hpp"
#include "fa/math.hpp"
#include <stdexcept>
#include <limits>

namespace fa {

void apply_logits_mask_inplace(float* /*row_logits*/, int /*N*/, const Tensor& /*mask_row*/, int /*key_row_index*/) {
    // Base commit: no-op. PR2 will implement masking logic.
}

} // namespace fa

namespace fa::mask {

void validate_padding_mask_b11n(const Tensor& Q, const Tensor& M) {
    if (M.ndim()!=4) throw std::invalid_argument("mask must be 4D (B,1,1,N)");
    if (M.dim(0)!=Q.dim(0)) throw std::invalid_argument("mask B mismatch");
    if (M.dim(1)!=1 || M.dim(2)!=1) throw std::invalid_argument("mask must be (B,1,1,N)");
    if (M.dim(3)!=Q.dim(2)) throw std::invalid_argument("mask N mismatch");
}

void apply_padding_mask_logits(std::vector<float>& logits, const Tensor& M, int b, int N) {
    const float ninf = fa::math::neg_inf();
    for (int j=0;j<N;++j) {
        const float keep = M.at(b,0,0,j);
        if (keep==0.0f) logits[j] = ninf; // zero means "masked"
    }
}

} // namespace fa::mask
