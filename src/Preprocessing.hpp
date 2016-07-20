#ifndef Preprocessing_hpp
#define Preprocessing_hpp

#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {

namespace preprocessing {

cv::Rect calc_horiz_roi(cv::Mat src);

} /* namespace preprocessing */

} /* namespace sonar_target_tracking */

#endif /* end of include guard: PreProcessing */
