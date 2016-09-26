#ifndef sonar_target_tracking_BasicOperations_hpp
#define sonar_target_tracking_BasicOperations_hpp

#include "sonar_target_tracking/SonarHolder.hpp"

namespace sonar_target_tracking {

namespace basic_operations {

void horizontal_sum(const SonarHolder& holder, int start_bin = 0);

} // namespace basic_operations
    
} // namespace sonar_target_tracking

#endif /* end of include guard: sonar_target_tracking_BasicOperations_hpp */
