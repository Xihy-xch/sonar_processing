#ifndef sonar_processing_Utilities_Hpp
#define sonar_processing_Utilities_Hpp

#include <cstdio>
#include <stdint.h>
#include <vector>
#include <sys/time.h>

namespace sonar_processing {

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

namespace now {

inline static uint64_t microseconds() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000000 + t.tv_usec;
}

inline static uint64_t milliseconds() {
    return microseconds() / 1000;    
}

}


} /* end of namespace utilities */

} /* end of namespace sonar_processing */

#endif /* end of include guard:  */
