#ifndef sonar_processing_ImageFiltering_hpp
#define sonar_processing_ImageFiltering_hpp

#include <iostream>
#include <opencv2/opencv.hpp>

namespace sonar_processing {

    namespace image_filtering {

        enum BorderFilterType {
            kSobel = 0,
            kSCharr,
            kPrewitt
        };

        void
        saliency_mapping(cv::InputArray src_arr, cv::OutputArray dst_arr, int block_count, cv::InputArray mask_arr);

        void
        integral_mean_filter(cv::InputArray integral_arr, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr);

        void
        mean_difference_filter(cv::InputArray src_arr0, cv::InputArray src_arr1, cv::OutputArray dst_arr, int ksize,
                               cv::InputArray mask_arr);

        void saliency_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr = cv::noArray());

        void border_filter(cv::InputArray src_arr, cv::OutputArray dst_arr);

        void border_filter(const cv::Mat &src, cv::Mat &dst, const cv::Mat mask, BorderFilterType type = kSobel);

        void mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize,
                         cv::InputArray mask_arr = cv::noArray());

        void meand_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize_inner, int ksize_outer,
                          cv::InputArray mask_arr = cv::noArray());

        void insonification_correction(const cv::Mat &src, const cv::Mat &mask, cv::Mat &dst);

        void filter2d(const cv::Mat &src, cv::Mat &dst, const cv::Mat kernel, const cv::Mat &mask);

        void border_filter_kernel(cv::Mat &kernel, int direction, BorderFilterType type = kSobel);

    } /* namespace image_filtering */

} /* sonar_processing image_util */

#endif /* ImageUtils_hpp */
