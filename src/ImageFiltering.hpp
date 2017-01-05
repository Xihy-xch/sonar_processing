#ifndef sonar_processing_ImageFiltering_hpp
#define sonar_processing_ImageFiltering_hpp

#include <iostream>
#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace image_filtering {

void saliency_mapping(cv::InputArray src_arr, cv::OutputArray dst_arr, int block_count, cv::InputArray mask_arr);

void integral_mean_filter(cv::InputArray integral_arr, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr);

void mean_difference_filter(cv::InputArray src_arr0, cv::InputArray src_arr1, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr);

void saliency_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr = cv::noArray());

void border_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr = cv::noArray());

void mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr = cv::noArray());

void meand_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize_inner, int ksize_outer, cv::InputArray mask_arr = cv::noArray());

// void minimum_blob_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size min_size, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);

} /* namespace image_filtering */

} /* sonar_processing image_util */

#endif /* ImageUtils_hpp */
