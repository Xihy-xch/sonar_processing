#ifndef Preprocessing_hpp
#define Preprocessing_hpp

#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {

namespace preprocessing {

cv::Rect calc_horiz_roi_old(cv::Mat src);

cv::Rect calc_horiz_roi(cv::Mat src);

double horiz_difference(cv::Mat src);
double vert_difference(cv::Mat src);

void mean_horiz_difference_threshold(cv::InputArray src, cv::OutputArray dst, uint32_t bsize,
                                double mean_thresh, double horiz_deriv_thresh);

void mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize);

void difference_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize);

std::vector<std::vector<cv::Point> > find_contours_and_filter(cv::Mat src, double area_factor, double width_factor, double height_factor);

std::vector<uint32_t> compute_ground_distance_line(cv::Mat mat);

cv::Mat remove_ground_distance(cv::Mat src, cv::Rect& horiz_roi);

cv::Mat remove_ground_distance_accurate(cv::Mat src, cv::Rect& horiz_roi);

uint32_t find_first_higher(cv::Mat mat, uint32_t row);

std::vector<double> background_features_estimation(cv::Mat mat, uint32_t bsize);

void background_features_difference(cv::InputArray src_arr, cv::OutputArray dst_arr, std::vector<double> features, uint32_t bsize);

std::vector<cv::Point> find_biggest_contour(cv::Mat src);

std::vector<std::vector<cv::Point> > target_detect_by_high_intensities(cv::InputArray src_arr);

void high_intensities_clustering(cv::InputArray src_arr);

std::vector<std::vector<cv::Point> > adaptative_target_detect_by_high_intensities(cv::InputArray src_arr);


} /* namespace preprocessing */

} /* namespace sonar_target_tracking */

#endif /* end of include guard: PreProcessing */
