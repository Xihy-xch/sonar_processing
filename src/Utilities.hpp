#ifndef sonar_target_tracking_Utilities_Hpp
#define sonar_target_tracking_Utilities_Hpp

#include <stdint.h>

namespace sonar_target_tracking {

namespace utilities {

static double clip(double val, double min, double max) {
    return (val < min) ? min : ((val > max) ? max : val);
}

static uint32_t border_fit(uint32_t x,  uint32_t total_size, uint32_t block_size) {
    return (total_size > x + block_size) ? x : total_size - block_size - 1;
}

} /* end of namespace utilities */

} /* end of namespace sonar_target_tracking */

#endif /* end of include guard:  */
