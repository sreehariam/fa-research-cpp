#include <stdexcept>
#include <string>
#include <vector>

namespace fa::checks {

inline void expect(bool cond, const std::string& msg) {
    if (!cond) throw std::invalid_argument(msg);
}

inline void equal_shape(const std::vector<int>& a, const std::vector<int>& b, const char* what) {
    if (a != b) throw std::invalid_argument(std::string("Shape mismatch for ") + what);
}

} // namespace fa::checks
