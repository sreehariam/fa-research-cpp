// include/fa/types.hpp
#pragma once

namespace fa {

struct AttentionOpts {
    bool  causal      = false;
    float temperature = 1.0f;  // PR4
};

} // namespace fa
