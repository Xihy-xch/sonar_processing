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

    std::vector<std::vector<cv::Point> > hi_contours = preprocessing::find_target_contours(mat);

    cv::Mat result = cv::Mat::zeros(cv::Size(bin_count_, beam_count_), CV_32F);

    for( int i = 0; i < hi_contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(hi_contours[i]) );
        src(cv::Rect(rc.x + horiz_roi_.x , rc.y, rc.width, rc.height)).copyTo(result(cv::Rect(rc.x + horiz_roi_.x, rc.y, rc.width, rc.height)));
    }

    polarshow("result data", image_utils::mat2vector<float>(result), bearings_, bin_count_, beam_count_);

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
