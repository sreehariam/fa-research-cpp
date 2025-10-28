#pragma once
#include <cstddef>

namespace fa::math {

float neg_inf();                           // -inf sentinel
float safe_exp(float v);                   // clamped exp for stability
float row_max(const float* x, int n);      // max over row of length n
float row_sumexp_stable(const float* x,
                        int n,
                        float m);          // sum(exp(x_i - m))

} // namespace fa::math
