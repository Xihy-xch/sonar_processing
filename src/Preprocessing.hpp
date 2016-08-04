#ifndef Preprocessing_hpp
#define Preprocessing_hpp

#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {

namespace preprocessing {

cv::Rect calc_horiz_roi_old(cv::Mat src);

cv::Rect calc_horiz_roi(cv::Mat src);

double horiz_deriv(cv::Mat src);

void adaptative_threshold(cv::InputArray src_arr, cv::OutputArray dst_arr);

void mean_horiz_deriv_threshold(cv::InputArray src, cv::OutputArray dst, uint32_t bsize,
                                double mean_thresh, double horiz_deriv_thresh);

std::vector<std::vector<cv::Point> > find_contours_and_filter(cv::Mat src, double area_factor, double width_factor, double height_factor);

std::vector<uint32_t> compute_ground_distance_line(cv::Mat mat);

cv::Mat remove_ground_distance(cv::Mat src, cv::Rect& horiz_roi);

cv::Mat remove_ground_distance_accurate(cv::Mat src, cv::Rect& horiz_roi);

uint32_t find_first_higher(cv::Mat mat, uint32_t row);

std::vector<double> background_features_estimation(cv::Mat mat, uint32_t bsize);

void background_features_difference(cv::InputArray src_arr, cv::OutputArray dst_arr, std::vector<double> features, uint32_t bsize);

std::vector<cv::Point> find_biggest_contour(cv::Mat src);


} /* namespace preprocessing */

} /* namespace sonar_target_tracking */

#endif /* end of include guard: PreProcessing */
