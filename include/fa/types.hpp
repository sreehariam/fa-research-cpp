// include/fa/types.hpp
#pragma once

namespace fa {

struct AttentionOpts {
    bool  causal      = false;
    float temperature = 1.0f; 
	float dropout_prob  = 0.0f;
};

} // namespace fa
