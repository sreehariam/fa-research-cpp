#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#include <initializer_list>
#include <cstddef>
#include <random>
#include <algorithm>
#include <numeric>

namespace fa {

// Simple owning float32 tensor. Contiguous storage, row-major.
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<int>& shape)
        : shape_(shape) {
        validate_shape();
        data_.assign(static_cast<size_t>(numel()), 0.0f);
        compute_strides();
    }

    static Tensor zeros(const std::vector<int>& shape) {
        Tensor t(shape);
        std::fill(t.data_.begin(), t.data_.end(), 0.0f);
        return t;
    }

    static Tensor randn(const std::vector<int>& shape, uint64_t seed) {
        Tensor t(shape);
        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& x : t.data_) x = dist(rng);
        return t;
    }

    int ndim() const { return static_cast<int>(shape_.size()); }
    int dim(int i) const { return shape_.at(i); }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }

    long long numel() const {
        long long n = 1;
        for (int s : shape_) n *= s;
        return n;
    }

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    // Flattened index access with bounds check.
    float& at_index(long long idx) {
        if (idx < 0 || idx >= numel()) throw std::out_of_range("Tensor index out of range");
        return data_[static_cast<size_t>(idx)];
    }
    const float& at_index(long long idx) const {
        if (idx < 0 || idx >= numel()) throw std::out_of_range("Tensor index out of range");
        return data_[static_cast<size_t>(idx)];
    }

    // 4D convenience accessor (B,H,N,D). Throws if ndim != 4.
    float& at(int b, int h, int n, int d) {
        require_ndim(4);
        long long idx = ((long long)b * strides_[0]) +
                        ((long long)h * strides_[1]) +
                        ((long long)n * strides_[2]) +
                        ((long long)d * strides_[3]);
        return at_index(idx);
    }
    const float& at(int b, int h, int n, int d) const {
        require_ndim(4);
        long long idx = ((long long)b * strides_[0]) +
                        ((long long)h * strides_[1]) +
                        ((long long)n * strides_[2]) +
                        ((long long)d * strides_[3]);
        return at_index(idx);
    }

    bool contiguous() const {
        std::vector<int> st(shape_.size());
        int nd = ndim();
        int stride = 1;
        for (int i = nd - 1; i >= 0; --i) {
            st[i] = stride;
            stride *= dim(i);
        }
        return st == strides_;
    }

private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    std::vector<float> data_;

    void validate_shape() const {
        if (shape_.empty()) throw std::invalid_argument("Tensor shape cannot be empty");
        for (int s : shape_) {
            if (s <= 0) throw std::invalid_argument("Tensor dims must be positive");
        }
    }

    void compute_strides() {
        strides_.assign(shape_.size(), 0);
        int nd = ndim();
        int stride = 1;
        for (int i = nd - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= dim(i);
        }
    }

    void require_ndim(int k) const {
        if (ndim() != k) throw std::invalid_argument("Tensor requires ndim=" + std::to_string(k));
    }
};

} // namespace fa
