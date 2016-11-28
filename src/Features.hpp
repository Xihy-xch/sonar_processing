#ifndef sonar_processing_features_hpp
#define sonar_processing_features_hpp

#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace features {

void saliency(cv::InputArray src_arr, cv::OutputArray dst_arr);

void saliency(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr);

void saliency_gray(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr);

void saliency_color(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr);

} /* namespace features */

} /* namespace sonar_processing */


#endif /* SaliencyMap_hpp */
