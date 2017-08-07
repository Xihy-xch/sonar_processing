#include "Clustering.hpp"
#include "Denoising.hpp"
#include "FrequencyDomain.hpp"
#include "ImageFiltering.hpp"
#include "ImageUtil.hpp"
#include "Preprocessing.hpp"
#include "ROI.hpp"
#include "SonarHolder.hpp"
#include "SonarImagePreprocessing.hpp"

namespace sonar_processing {

SonarImagePreprocessing::SonarImagePreprocessing()
    : mean_filter_ksize_(7)
    , mean_difference_filter_ksize_(25)
    , median_blur_filter_ksize_(5)
{
}

SonarImagePreprocessing::~SonarImagePreprocessing() {
}

void SonarImagePreprocessing::ExtractROI(
    const SonarHolder& sonar_holder,
    cv::Mat& roi_cart,
    uint32_t& roi_line,
    float alpha,
    int start_row,
    int end_row) const
{
    uint32_t bin_count = sonar_holder.bin_count();
    uint32_t beam_count = sonar_holder.beam_count();

    cv::Mat mask = sonar_holder.cart_image_mask();
    cv::Mat sonar_image = sonar_holder.cart_image();

    if (end_row<0) end_row=sonar_image.rows;

    // calculate the proportional mean of each image row
    std::vector<float> row_mean(end_row, 0);
    for (size_t i = start_row; i <= end_row; i++) {
        int r = sonar_image.rows-i-1;
        double value = cv::sum(sonar_image.row(r))[0] / cv::countNonZero(mask.row(r));
        row_mean[i] = std::isnan(value) ? 0 : value;
    }

    // accumulative sum
    std::vector<float> accum_sum(row_mean.size(), 0);
    std::partial_sum(row_mean.begin(), row_mean.end(), accum_sum.begin());

    // threshold
    float min = *std::min_element(accum_sum.begin(), accum_sum.end());
    float max = *std::max_element(accum_sum.begin(), accum_sum.end());
    float thresh = alpha * (max - min) + min;
    std::replace_if(accum_sum.begin(), accum_sum.end(), std::bind2nd(std::less<float>(), thresh), 0.0);

    // generate new cartesian mask
    std::vector<float>::iterator pos = std::find_if (accum_sum.begin(), accum_sum.end(), std::bind2nd(std::greater<float>(), 0));
    uint32_t new_y = std::distance(accum_sum.begin(), pos) + 1;
    roi_line = mask.rows - new_y;
    mask(cv::Rect(0, mask.rows - new_y, mask.cols, new_y)).setTo(cv::Scalar(0));
    mask.copyTo(roi_cart);
}

void SonarImagePreprocessing::Apply(
    const SonarHolder& sonar_holder,
    cv::Mat& preprocessed_image,
    cv::Mat& result_mask,
    float scale_factor) const
{
    cv::Mat roi_cart;
    uint32_t roi_line;
    ExtractROI(sonar_holder, roi_cart, roi_line, 0.005, 30, sonar_holder.cart_size().height-1);
    Apply(sonar_holder.cart_image(), roi_cart, preprocessed_image, result_mask, scale_factor, roi_line);
}

void SonarImagePreprocessing::Apply(
    const cv::Mat& source_cart_image,
    const cv::Mat& source_cart_mask,
    cv::Mat& preprocessed_image,
    cv::Mat& result_mask,
    float scale_factor,
    int start_cart_line) const
{

    cv::Mat cart_image = source_cart_image;
    cv::Mat cart_mask = source_cart_mask;

    if (scale_factor != 1.0) {
        cv::Size new_size = cv::Size(cart_image.size().width*scale_factor, cart_image.size().height*scale_factor);
        cv::resize(cart_image, cart_image, new_size);
        cv::resize(cart_mask, cart_mask, new_size);
    }

    cv::Mat cart_image_8u;
    cart_image.convertTo(cart_image_8u, CV_8U, 255);

    // apply insonification correction
    cv::Mat enhanced;
    image_filtering::insonification_correction(cart_image, cart_mask, enhanced);

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

    preprocessed_image = cv::Mat::zeros(mean_diff.size(), mean_diff.type());
    cv::normalize(mean_diff, preprocessed_image, 0, 1, cv::NORM_MINMAX, CV_32FC1, cart_mask);

    cart_mask.copyTo(result_mask);

    if (scale_factor != 1.0) {
        cv::resize(preprocessed_image, preprocessed_image, source_cart_image.size());
        cv::resize(result_mask, result_mask, source_cart_image.size());
    }
}


} /* namespace sonar_processing */
