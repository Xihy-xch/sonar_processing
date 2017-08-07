#ifndef sonar_processing_SonarImagePreprocessing_hpp
#define sonar_processing_SonarImagePreprocessing_hpp

#include <iostream>
#include <numeric>
#include "SonarHolder.hpp"

namespace sonar_processing {

class SonarImagePreprocessing {

public:
    SonarImagePreprocessing();
    ~SonarImagePreprocessing();

    void Apply(
        const SonarHolder& sonar_holder,
        cv::Mat& preprocessed_image,
        cv::Mat& result_mask,
        float scale_factor=1.0) const;

    void Apply(
        const cv::Mat& source_cart_image,
        const cv::Mat& source_cart_mask,
        cv::Mat& preprocessed_image,
        cv::Mat& result_mask,
        float scale_factor=1.0,
        int start_cart_line=0) const;

    void ExtractROI(
        const SonarHolder& sonar_holder,
        cv::Mat& roi_cart,
        uint32_t& roi_line,
        float alpha,
        int start_row=0,
        int end_row=-1) const;

    void set_mean_filter_ksize(int mean_filter_ksize) {
        mean_filter_ksize_ = mean_filter_ksize;
    }

    int mean_filter_ksize() const {
        return mean_filter_ksize_;
    }

    void set_mean_difference_filter_ksize(int mean_difference_filter_ksize) {
        mean_difference_filter_ksize_ = mean_difference_filter_ksize;
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

private:

    // denoising mean filter kernel size
    int mean_filter_ksize_;

    // mean difference filter kernel size
    int mean_difference_filter_ksize_;

    // median blur filter kernel size;
    int median_blur_filter_ksize_;

};

} /* namespace sonar_processing*/


#endif /* SonarPreprocessing_hpp */
