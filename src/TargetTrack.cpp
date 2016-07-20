#include "rock_util/SonarSampleConverter.hpp"
#include "sonar_target_tracking/TargetTrack.hpp"
#include "sonar_target_tracking/ImageUtils.hpp"
#include "sonar_target_tracking/Preprocessing.hpp"

namespace sonar_target_tracking 
{

using namespace image_utils;

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

    cv::Mat mat = cv::Mat(bins_).reshape(1, beam_count_);
    cv::Mat mat_eqhist = cv::Mat::zeros(mat.size(), mat.type());
    image_utils::cv32f_equalize_histogram(mat, mat_eqhist);
    cv::medianBlur(mat_eqhist, mat_eqhist, 5);

    horiz_roi_ = preprocessing::calc_horiz_roi(mat_eqhist);
}

}
