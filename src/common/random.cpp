#include <random>

namespace fa::rnd {

inline void fill_randn(float* data, long long n, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (long long i = 0; i < n; ++i) data[i] = dist(rng);
}

} // namespace fa::rnd
