#pragma once
namespace fa {

struct AttentionOpts {
    bool causal = false;
    float dropout = 0.0f; // must be 0 in base; future PRs may implement
    int block_m = 64;     // reserved for tiled variants
    int block_n = 64;
};

} // namespace fa
