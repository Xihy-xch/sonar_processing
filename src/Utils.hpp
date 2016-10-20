#ifndef sonar_target_tracking_Utilities_Hpp
#define sonar_target_tracking_Utilities_Hpp

#include <stdint.h>
#include <vector>

namespace sonar_target_tracking {

namespace utils {

inline double clip(double val, double min, double max) {
    return (val < min) ? min : ((val > max) ? max : val);
}

inline uint32_t border_fit(uint32_t x,  uint32_t total_size, uint32_t block_size) {
    return (total_size > x + block_size) ? x : total_size - block_size - 1;
}

template <typename T>
void accumulative_sum(const std::vector<T>& src, std::vector<T>& dst) {
    dst.resize(src.size());
    dst[0] = src[0];
    for (size_t i = 1; i < src.size(); i++) dst[i] = dst[i-1] + src[i];
}

} /* end of namespace utilities */

} /* end of namespace sonar_target_tracking */

#endif /* end of include guard:  */
