#include "base/Plot.hpp"
#include "base/MathUtil.hpp"
#include "sonar_util/Plot.hpp"
#include "rock_util/SonarSampleConverter.hpp"
#include "sonar_target_tracking/TargetTrack.hpp"
#include "sonar_target_tracking/ImageUtils.hpp"
#include "sonar_target_tracking/Preprocessing.hpp"
#include "sonar_target_tracking/third_party/spline.h"

using namespace sonar_util::plot;

namespace sonar_target_tracking
{

TargetTrack::TargetTrack(
    const std::vector<float>& bins,
    const std::vector<float>& bearings,
    uint32_t beam_count,
    uint32_t bin_count)
    : bins_(bins)
    , bearings_(bearings)
    , beam_count_(beam_count)
    , bin_count_(bin_count) {
}

void TargetTrack::apply() {
    cv::Mat src = cv::Mat(bins_).reshape(1, beam_count_);
    polarshow("source data", bins_, bearings_, bin_count_, beam_count_);

    cv::Mat mat = preprocessing::remove_ground_distance(src, horiz_roi_);
    detect_target_bounding_rect(mat);
}

void TargetTrack::remove_background(std::vector<double> features, uint32_t bsize) {

    cv::Mat src = cv::Mat(bins_).reshape(1, beam_count_);
    cv::Mat mat = preprocessing::remove_ground_distance_accurate(src, horiz_roi_);

    image_utils::cv32f_equalize_histogram(mat, mat);
    image_utils::cv32f_clahe(mat, mat);

    cv::Mat diff_values;
    preprocessing::background_features_difference(mat, diff_values, features, bsize);
}

void TargetTrack::detect_target_bounding_rect(cv::Mat src) {

    cv::Mat bin;
    preprocessing::adaptative_threshold(src, bin);

    // cv::Mat bin;
    // cv::normalize(src, bin, 0, 1, cv::NORM_MINMAX);
    // cv::boxFilter(bin, bin, CV_32F, cv::Size(5, 5));
    // cv::threshold(bin, bin, 0.7, 1.0, cv::THRESH_BINARY);
    //
    // uint32_t bsize = 10;
    // cv::Mat mat;
    // preprocessing::mean_horiz_deriv_threshold(bin, mat, bsize, 0.3, 0.2);
    //
    // mat.convertTo(mat, CV_8UC1, 255);
    // std::vector<std::vector<cv::Point> > contours;
    //
    // contours = preprocessing::find_contours_and_filter(mat, cv::Size(mat.cols * 0.05, mat.rows * 0.05));
    //
    // mat.setTo(0);
    // cv::drawContours(mat, contours, -1, cv::Scalar(255), CV_FILLED);
    //
    // cv::morphologyEx(mat, mat, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    // cv::morphologyEx(mat, mat, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    //
    // contours = preprocessing::find_contours_and_filter(mat, cv::Size(mat.cols * 0.15, mat.rows * 0.15));
    //
    // int offset_x = bin_count_ - src.cols;
    //
    // mat = cv::Mat::zeros(cv::Size(bin_count_, beam_count_), CV_32FC1);
    // cv::Mat im_sonar = cv::Mat(bins_).reshape(1, beam_count_);
    //
    // for( int i = 0; i < contours.size(); i++ ) {
    //     cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
    //     im_sonar(cv::Rect(rc.x + offset_x, rc.y, rc.width, rc.height)).copyTo(mat(cv::Rect(rc.x + offset_x, rc.y, rc.width, rc.height)));
    // }
    //
    // polarshow("result data", image_utils::mat2vector<float>(mat), bearings_, bin_count_, beam_count_);
}

}
