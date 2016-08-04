#ifndef sonar_target_tracking_Utilities_Hpp
#define sonar_target_tracking_Utilities_Hpp

namespace sonar_target_tracking {

namespace utilities {

double clip(double val, double min, double max) {
    return (val < min) ? min : ((val > max) ? max : val);
}

} /* end of namespace utilities */

} /* end of namespace sonar_target_tracking */

#endif /* end of include guard:  */
