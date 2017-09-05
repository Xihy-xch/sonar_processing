#ifndef sonar_processing_SonarImagePreprocessing_hpp
#define sonar_processing_SonarImagePreprocessing_hpp

#include <iostream>
#include <numeric>
#include "ImageFiltering.hpp"
#include "SonarHolder.hpp"

namespace sonar_processing {

class SonarImagePreprocessing {

public:

    enum MeanDifferenceFilterSource {
        kBorder = 0,
        kEnhanced = 1
    };

    SonarImagePreprocessing();
    ~SonarImagePreprocessing();

    void Apply(
        const SonarHolder& sonar_holder,
        cv::Mat& preprocessed_image,
        cv::Mat& result_mask,
        float scale_factor=1.0) const;

    void Apply(
        const cv::Mat& source_image,
        const cv::Mat& source_mask,
        cv::Mat& preprocessed_image,
        cv::Mat& result_mask,
        float scale_factor=1.0) const;

    void set_mean_filter_ksize(int mean_filter_ksize) {
        mean_filter_ksize_ = mean_filter_ksize;
    }

    int mean_filter_ksize() const {
        return mean_filter_ksize_;
    }

    void set_mean_difference_filter_ksize(int mean_difference_filter_ksize) {
        mean_difference_filter_ksize_ = mean_difference_filter_ksize;
    }

    void set_mean_difference_filter_enable(bool mean_difference_filter_enable) {
        mean_difference_filter_enable_ = mean_difference_filter_enable;
    }

    bool mean_difference_filter_enable() const {
        return mean_difference_filter_enable_;
    }

    int mean_difference_filter_ksize() const {
        return mean_difference_filter_ksize_;
    }

    void set_median_blur_filter_ksize(int median_blur_filter_ksize) {
        median_blur_filter_ksize_ = median_blur_filter_ksize;
    }

    int median_blur_filter_ksize() const {
        return median_blur_filter_ksize_;
    }

    void set_roi_extract_thresh(double roi_extract_thresh) {
        roi_extract_thresh_ = roi_extract_thresh;
    }

    double roi_extract_thresh() const {
        return roi_extract_thresh_;
    }

    void set_roi_extract_start_bin(int roi_extract_start_bin) {
        roi_extract_start_bin_ = roi_extract_start_bin;
    }

    int roi_extract_start_bin() const {
        return roi_extract_start_bin_;
    }

    void set_border_filter_type(image_filtering::BorderFilterType border_filter_type) {
        border_filter_type_ = border_filter_type;
    }

    image_filtering::BorderFilterType border_filter_type() const {
        return border_filter_type_;
    }

    void set_border_filter_enable(bool border_filter_enable){
        border_filter_enable_ = border_filter_enable;
    }

    bool border_filter_enable() const {
        return border_filter_enable_;
    }

    void set_mean_difference_filter_source(MeanDifferenceFilterSource mean_difference_filter_source) {
        mean_difference_filter_source_ = mean_difference_filter_source;
    }

    MeanDifferenceFilterSource mean_difference_filter_source() const {
        return mean_difference_filter_source_;
    }

private:

    void PerformPreprocessing(
        const cv::Mat& source_cart_image,
        const cv::Mat& source_cart_mask,
        cv::Mat& preprocessed_image,
        cv::Mat& result_mask,
        float scale_factor=1.0,
        int start_cart_line=0) const;

    void ExtractROI(
        const cv::Mat& source_image,
        const cv::Mat& source_mask,
        cv::Mat& roi_cart,
        uint32_t& roi_line,
        float alpha,
        int start_row=0,
        int end_row=-1) const;

    // denoising mean filter kernel size
    int mean_filter_ksize_;

    // mean difference filter kernel size
    int mean_difference_filter_ksize_;

    // median blur filter kernel size;
    int median_blur_filter_ksize_;

    // the roi extract threshold
    double roi_extract_thresh_;

    /// the roi extract start bin
    int roi_extract_start_bin_;

    // enable / disable mean difference filter
    bool mean_difference_filter_enable_;

    // the border filter type
    image_filtering::BorderFilterType border_filter_type_;

    // enable / disable border filter
    bool border_filter_enable_;

    MeanDifferenceFilterSource mean_difference_filter_source_;

};

} /* namespace sonar_processing*/


#endif /* SonarPreprocessing_hpp */
