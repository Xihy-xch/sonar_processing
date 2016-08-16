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

    horiz_roi_ = preprocessing::calc_horiz_roi(src, 0.075);
    cv::Mat mat = src(horiz_roi_);
    mat.convertTo(mat, CV_8U, 255);

    preprocessing::remove_low_intensities_columns(mat, mat);

    std::vector<std::vector<cv::Point> > hi_contours = preprocessing::find_target_contours(mat);
    std::vector<std::vector<cv::Point> > shadow_contours = preprocessing::find_shadow_contours(mat);

    uint32_t frame_height = 512;
    float angle = bearings_[bearings_.size()-1];
    uint32_t frame_width = base::MathUtil::aspect_ratio_width(angle, frame_height);
    std::vector<int> beam_mapping = sonar_util::Converter::generate_beam_mapping_from_cartesian(bins_, bearings_, bin_count_, beam_count_, frame_width, frame_height);
    
    
    cv::Mat hi_mask = cv::Mat::zeros(cv::Size(bin_count_, beam_count_), CV_8UC1);
    cv::Mat shadow_mask = cv::Mat::zeros(cv::Size(bin_count_, beam_count_), CV_8UC1);
    cv::Mat result = cv::Mat::zeros(cv::Size(bin_count_, beam_count_), CV_32F);
    
    for( int i = 0; i < hi_contours.size(); i++ ) {
        for (int j = 0; j < hi_contours[i].size(); j++){
            hi_contours[i][j].x += horiz_roi_.x;
        }
    }
    
    for( int i = 0; i < shadow_contours.size(); i++ ) {
        for (int j = 0; j < shadow_contours[i].size(); j++){
            shadow_contours[i][j].x += horiz_roi_.x;
        }
    }

    cv::drawContours(hi_mask, hi_contours, -1, cv::Scalar(255), CV_FILLED);
    cv::drawContours(shadow_mask, shadow_contours, -1, cv::Scalar(255), CV_FILLED);

    hi_mask.convertTo(hi_mask, CV_32F, 1.0/255.0);
    shadow_mask.convertTo(shadow_mask, CV_32F, 1.0/255.0);
    
    cv::Mat hi_mask_polar = sonar_util::Converter::convert2polar(image_utils::mat2vector<float>(hi_mask), bearings_, bin_count_, beam_count_, frame_width, frame_height, beam_mapping);
    cv::Mat shadow_mask_polar = sonar_util::Converter::convert2polar(image_utils::mat2vector<float>(shadow_mask), bearings_, bin_count_, beam_count_, frame_width, frame_height, beam_mapping);
    cv::Mat src_polar = sonar_util::Converter::convert2polar(bins_, bearings_, bin_count_, beam_count_, frame_width, frame_height, beam_mapping);
    
    cv::Mat canvas_polar;
    cv::cvtColor(src_polar, canvas_polar, CV_GRAY2BGR);
    
    std::vector<std::vector<cv::Point> > hi_contours_polar = preprocessing::find_contours(hi_mask_polar, CV_RETR_EXTERNAL, true);
    std::vector<std::vector<cv::Point> > shadow_contours_polar = preprocessing::find_contours(shadow_mask_polar, CV_RETR_EXTERNAL);
    cv::drawContours(canvas_polar, hi_contours_polar, -1, cv::Scalar(0, 0, 255), 2);
    cv::drawContours(canvas_polar, shadow_contours_polar, -1, cv::Scalar(255, 0, 0), 2);
    
    cv::imshow("canvas_polar", canvas_polar);
}

void TargetTrack::remove_background(std::vector<double> features, uint32_t bsize) {

    cv::Mat src = cv::Mat(bins_).reshape(1, beam_count_);
    cv::Mat mat = preprocessing::remove_ground_distance_accurate(src, horiz_roi_);

    image_utils::equalize_histogram_32f(mat, mat);
    image_utils::clahe_32f(mat, mat);

    cv::Mat diff_values;
    preprocessing::background_features_difference(mat, diff_values, features, bsize);
}

}
