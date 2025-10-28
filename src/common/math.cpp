#include "fa/math.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace fa::math {

float neg_inf() {
    return -std::numeric_limits<float>::infinity();
}

float safe_exp(float v) {
    if (v < -80.0f) return 0.0f;  // avoid underflow/denorms
    if (v > 80.0f)  v = 80.0f;
    return std::exp(v);
}

float row_max(const float* x, int n) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) m = std::max(m, x[i]);
    return m;
}

float row_sumexp_stable(const float* x, int n, float m) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += safe_exp(x[i] - m);
    return s;
}

} // namespace fa::math
