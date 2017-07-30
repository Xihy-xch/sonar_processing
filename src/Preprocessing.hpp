#ifndef Preprocessing_hpp
#define Preprocessing_hpp

#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace preprocessing {

static void slide_window_base(cv::InputArray src_arr) {
    cv::Mat src = src_arr.getMat();

    uint32_t bsize = 32;
    uint32_t bstep = bsize;

    cv::Mat src_canvas = cv::Mat::zeros(src.size(), CV_8UC3);

    for (int y = 0; y < src.cols-bsize; y+=bstep) {
        for (int x = 0; x < src.cols-bsize; x+=bstep) {
            cv::cvtColor(src, src_canvas, CV_GRAY2BGR);
            cv::Rect roi = cv::Rect(x, y, bsize, bsize);

            cv::rectangle(src_canvas, roi, cv::Scalar(0, 0, 255));

            cv::imshow("src_canvas", src_canvas);
            if ((char)cv::waitKey() == 27) {
                return;
            }
        }
    }
}


template <typename T>
static void min_max_element(std::vector<T> v, T& min, T& max) {
    min = *std::min_element(v.begin(), v.end());
    max = *std::max_element(v.begin(), v.end());
}

template <typename T>
static T min_max_thresh(std::vector<T> v, double alpha) {
    double max, min;
    min_max_element<double>(v, min, max);
    return alpha * (max - min) + min;
}


cv::Rect calc_horiz_roi_old(cv::Mat src);

cv::Rect calc_horiz_roi(cv::Mat src, float alpha = 0.2);

void adaptive_clahe(cv::InputArray _src, cv::OutputArray _dst, unsigned int max_clip_limit = 20);

double horiz_difference(cv::Mat src);

double vert_difference(cv::Mat src);

void weak_target_thresholding_old(cv::InputArray src_arr, cv::InputArray src_hc_arr, cv::OutputArray dst_arr);

void weak_target_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr);

void weak_shadow_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr);

void mean_horiz_difference_thresholding(cv::InputArray src, cv::OutputArray dst, uint32_t bsize,
                                double mean_thresh, double horiz_deriv_thresh);

void mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize);

void difference_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize);

void gradient_filter(cv::InputArray src_arr, cv::OutputArray dst_arr);

void contrast_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int div_size = 2);

void remove_low_intensities_columns(cv::InputArray src_arr, cv::OutputArray dst_arr);

std::vector<std::vector<cv::Point> > find_contours(cv::Mat src, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);

std::vector<std::vector<cv::Point> > find_contours_and_filter(cv::Mat src, cv::Size min_size, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);

std::vector<std::vector<cv::Point> > adaptative_find_contours_and_filter(cv::Mat src, double area_factor, double width_factor, double height_factor);

std::vector<cv::Point> compute_ground_distance_line(cv::Mat mat, float thresh_factor = 1.0);

cv::Mat remove_ground_distance(cv::Mat src, cv::Rect& horiz_roi);

cv::Mat remove_ground_distance_accurate(cv::Mat src, cv::Rect& horiz_roi);

uint32_t find_first_higher(cv::Mat mat, uint32_t row);

std::vector<double> background_features_estimation(cv::Mat mat, uint32_t bsize);

void background_features_difference(cv::InputArray src_arr, cv::OutputArray dst_arr, std::vector<double> features, uint32_t bsize);

float calc_spatial_variation_coefficient(std::vector<float> classes_val);
float spatial_variation_coefficient(cv::Mat src);
void spatial_variation_coefficient_filter(cv::InputArray src_arr, cv::OutputArray dst_arr);

void difference_of_gaussian(cv::InputArray src_arr, cv::OutputArray dst_arr);

std::vector<std::vector<cv::Point> > find_shadow_contours(cv::InputArray src_arr);

std::vector<std::vector<cv::Point> > find_target_contours(cv::InputArray src_arr);

std::vector<std::vector<cv::Point> > target_detect_by_high_intensities(cv::InputArray src_arr);

void simple_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr, double alpha = 0.3, uint32_t colsdiv = 2, cv::InputArray mask=cv::noArray());

void houghlines_mask(cv::InputArray src_arr, cv::OutputArray dst_arr,
                    double rho = 1.0, double theta = CV_PI/180.0, int threshold = 10,
                    double min_line_length = 20, double max_line_gap = 40);

void remove_blobs(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size min_size, int mode = CV_RETR_EXTERNAL, bool convex_hull = false);

std::vector<std::vector<cv::Point> > remove_low_intensities_contours(cv::InputArray src_arr, std::vector<std::vector<cv::Point> > contours);

std::vector<std::vector<cv::Point> > convexhull(std::vector<std::vector<cv::Point> > contours);

cv::Mat get_sonar_mask( const std::vector<float>& bearings, uint32_t bin_count, uint32_t beam_count, uint32_t frame_width, uint32_t frame_height);

void extract_roi_masks (const cv::Mat& cart_image,
                        const std::vector<float>& bearings,
                        uint32_t beam_count, uint32_t bin_count,
                        cv::Mat& roi_cartesian, cv::Mat& roi_polar,
                        float alpha = 0.05);

cv::Mat extract_roi_mask (const cv::Mat& sonar_image, cv::Mat mask,
                          const std::vector<float>& bearings,
                          uint32_t bin_count, uint32_t beam_count,
                          float alpha, bool cartesian = true);

cv::Mat extract_cartesian_mask (const cv::Mat& sonar_image, const cv::Mat& mask,
                        float alpha);

cv::Mat extract_cartesian_mask2(const cv::Mat& sonar_image, const cv::Mat& mask, float alpha);

std::vector<cv::Point> find_biggest_contour(cv::InputArray src_arr);

} /* namespace preprocessing */

} /* namespace sonar_processing */

#endif /* end of include guard: PreProcessing */
