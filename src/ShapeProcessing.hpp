#ifndef sonar_processing_ImageProcessing_hpp
#define sonar_processing_ImageProcessing_hpp

#include <vector>
#include <opencv2/opencv.hpp>
#include <climits>

namespace sonar_processing {

namespace shape_processing {

std::vector<std::vector<cv::Point> > find_contours(cv::InputArray src_arr, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);

std::vector<std::vector<cv::Point> > find_contours(cv::InputArray src_arr, cv::Size min_size, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);


} /* namespace shape_processing */

} /* namespace sonar_processing */

#endif /* ImageProcessing_hpp */
