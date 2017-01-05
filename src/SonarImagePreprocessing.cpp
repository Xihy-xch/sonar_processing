#include "base/MathUtil.hpp"
#include "sonar_processing/Clustering.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/FrequencyDomain.hpp"
#include "sonar_processing/ImageFiltering.hpp"
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/Preprocessing.hpp"
#include "sonar_processing/ROI.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "SonarImagePreprocessing.hpp"

namespace sonar_processing {

SonarImagePreprocessing::SonarImagePreprocessing()
    : clahe_final_clip_limit_(3.0)
    , mean_filter_ksize_(7)
    , mean_difference_filter_ksize_(50)
    , median_blur_filter_ksize_(5)
    , saliency_map_scale_factor_(0.25)
    , saliency_map_block_count_(8)
    , saliency_map_thresh_factor_(0.5)
    , distance_transform_thresh_(0.25)
{
}

SonarImagePreprocessing::~SonarImagePreprocessing() {
}

void SonarImagePreprocessing::Apply(const SonarHolder& sonar_holder, cv::OutputArray preprocessed_image, cv::OutputArray result_mask, float scale_factor) const {

    // get region of intereset mask
    int sb = roi::cartesian::find_start_bin(sonar_holder);
    cv::Mat raw_mask;
    sonar_holder.load_roi_mask(sb, raw_mask);

    cv::Mat cart_mask;
    sonar_holder.load_cartesian_image(raw_mask, cart_mask);
    cart_mask.convertTo(cart_mask, CV_8U, 255.0);

    int start_cart_line = (int)sonar_holder.cart_center_point(sb, sonar_holder.beam_count()/2).y;
    Apply(sonar_holder.cart_image(), cart_mask, preprocessed_image, result_mask, scale_factor, start_cart_line);
}

void SonarImagePreprocessing::Apply(
    const cv::Mat& source_cart_image, const cv::Mat& source_cart_mask,
    cv::OutputArray preprocessed_image, cv::OutputArray result_mask, float scale_factor, int start_cart_line) const {

    cv::Mat cart_image = source_cart_image;
    cv::Mat cart_mask = source_cart_mask;

    if (scale_factor != 1.0) {
        cv::Size new_size = cv::Size(cart_image.size().width*scale_factor, cart_image.size().height*scale_factor);
        cv::resize(cart_image, cart_image, new_size);
        cv::resize(cart_mask, cart_mask, new_size);
    }

    cv::Mat cart_image_8u;
    cart_image.convertTo(cart_image_8u, CV_8U, 255);

    cv::Mat enhanced;
    image_util::adaptative_clahe(cart_image_8u, enhanced, cv::Size(8, 8), 7.5, clahe_final_clip_limit_);
    enhanced.convertTo(enhanced, CV_32F, 1.0/255.0);

    // image denoising
    cv::Mat denoised;
    image_filtering::mean_filter(enhanced, denoised, mean_filter_ksize_, cart_mask);

    // apply border filter
    cv::Mat border, denoised_8u;
    denoised.convertTo(denoised_8u, CV_8U, 255.0);
    image_filtering::border_filter(denoised_8u, border);

    // reduce mask size
    image_util::erode(cart_mask, cart_mask, cv::Size(15, 15), 2);

    // apply cartesian mask
    image_util::apply_mask(border, border, cart_mask);
    border.convertTo(border, CV_32F, 1.0/255.0);
    cv::normalize(border, border, 0, 1, cv::NORM_MINMAX, CV_32FC1, cart_mask);

    // mean difference filter
    cv::Mat mean_diff;
    image_filtering::mean_difference_filter(enhanced, border, mean_diff, mean_difference_filter_ksize_, cart_mask);

    // apply median filter
    mean_diff.convertTo(mean_diff, CV_8U, 255.0);
    cv::medianBlur(mean_diff, mean_diff, median_blur_filter_ksize_);
    mean_diff.convertTo(mean_diff, CV_32F, 1.0/255.0);

    int h_padding = mean_diff.size().height-(start_cart_line*scale_factor);
    image_util::horizontal_normalize(mean_diff, mean_diff, cart_mask, h_padding);

    // compute saliency mapping
    cv::Mat saliency;
    cv::Mat reduce;
    cv::Size saliency_size = cv::Size(mean_diff.size().width*saliency_map_scale_factor_, mean_diff.size().height*saliency_map_scale_factor_);
    cv::resize(mean_diff, reduce, saliency_size);
    image_filtering::saliency_mapping(reduce, saliency, saliency_map_block_count_, cart_mask);
    cv::resize(saliency, saliency, mean_diff.size());
    cv::normalize(saliency, saliency, 0, 1, cv::NORM_MINMAX, CV_32FC1, cart_mask);

    // create saliency mask
    cv::Mat saliency_mask;
    float thresh = image_util::otsu_thresh_32f(saliency, cart_mask);
    cv::threshold(saliency, saliency_mask, thresh*saliency_map_thresh_factor_, 1.0, cv::NORM_MINMAX);

    // find biggest contour from saliency mask
    std::vector<std::vector<cv::Point> > contours;
    contours.push_back(preprocessing::find_biggest_contour(saliency_mask));

    // draw convex mask
    cv::Mat convex_mask = cv::Mat::zeros(saliency_mask.size(), CV_8UC1);
    cv::drawContours(convex_mask, contours, -1, cv::Scalar(255), CV_FILLED);

    // compute the distance transform
    cv::Mat distance;
    cv::distanceTransform(convex_mask, distance, CV_DIST_C, 0);
    cv::normalize(distance, distance, 0, 1, cv::NORM_MINMAX);

    // create a mask from distance
    cv::Mat distance_mask;
    cv::threshold(distance, distance_mask, distance_transform_thresh_, 255.0, cv::NORM_MINMAX);

    //increse distance mask size
    image_util::dilate(distance_mask, distance_mask, cv::Size(15, 15), 2);

    // apply saliency mask
    image_util::apply_mask(distance_mask, distance_mask, cart_mask);
    distance_mask.convertTo(distance_mask, CV_32F, 1.0/255.0);
    mean_diff = mean_diff.mul(distance_mask);
    cv::normalize(mean_diff, preprocessed_image, 0, 1, cv::NORM_MINMAX, CV_32FC1, cart_mask);
    distance_mask.copyTo(result_mask);

    if (scale_factor != 1.0) {
        cv::resize(preprocessed_image, preprocessed_image, source_cart_image.size());
        cv::resize(result_mask, result_mask, source_cart_image.size());
    }
}


} /* namespace sonar_processing */
