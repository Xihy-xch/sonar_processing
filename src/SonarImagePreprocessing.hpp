#ifndef sonar_processing_SonarImagePreprocessing_hpp
#define sonar_processing_SonarImagePreprocessing_hpp

#include <iostream>
#include "sonar_processing/SonarHolder.hpp"

namespace sonar_processing {

class SonarImagePreprocessing {

public:
    SonarImagePreprocessing();
    ~SonarImagePreprocessing();

    void Apply(const SonarHolder& sonar_holder, cv::OutputArray preprocessed_image, cv::OutputArray result_mask, float scale_factor=1.0) const;

    void Apply(const cv::Mat& source_cart_image, const cv::Mat& source_cart_mask, cv::OutputArray preprocessed_image, cv::OutputArray result_mask, float scale_factor=1.0, int start_cart_line=0) const;

    void set_clahe_final_clip_limit(float clahe_final_clip_limit) {
        clahe_final_clip_limit_ = clahe_final_clip_limit;
    }

    float clahe_final_clip_limit() const {
        return clahe_final_clip_limit_;
    }

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

    void set_saliency_map_scale_factor(float saliency_map_scale_factor) {
        saliency_map_scale_factor_ = saliency_map_scale_factor;
    }

    float saliency_map_scale_factor() const {
        return saliency_map_scale_factor_;
    }

    void set_saliency_map_block_count(int saliency_map_block_count) {
        saliency_map_block_count_ = saliency_map_block_count;
    }

    int saliency_map_block_count() const {
        return saliency_map_block_count_;
    }

    void set_saliency_map_thresh_factor(float saliency_map_thresh_factor) {
        saliency_map_thresh_factor_ = saliency_map_thresh_factor;
    }

    float saliency_map_thresh_factor() const {
        return saliency_map_thresh_factor_;
    }

    void set_distance_transform_thresh(float distance_transform_thresh) {
        distance_transform_thresh_ = distance_transform_thresh;
    }

    float distance_transform_thresh() const {
        return distance_transform_thresh_;
    }

private:

    // enhancement clahe final clip limit
    float clahe_final_clip_limit_;

    // denoising mean filter kernel size
    int mean_filter_ksize_;

    // mean difference filter kernel size
    int mean_difference_filter_ksize_;

    // median blur filter kernel size;
    int median_blur_filter_ksize_;

    // saliency map scale factor
    float saliency_map_scale_factor_;

    // saliency map block size
    int saliency_map_block_count_;

    // saliency map binary threshold factor adjust
    float saliency_map_thresh_factor_;

    // distance transform threshold
    float distance_transform_thresh_;
};

} /* namespace sonar_processing*/


#endif /* SonarPreprocessing_hpp */
