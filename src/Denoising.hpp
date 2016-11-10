#ifndef Denoising_hpp
#define Denoising_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <deque>

namespace sonar_processing {

namespace denoising {

void homomorphic_filter(cv::InputArray _src, cv::OutputArray _dst, int iterations);

void rls(cv::Mat& rls_w, cv::Mat& rls_p, cv::Mat& src, cv::Mat& dst);

void rls_sliding_window(cv::Mat& rls_w, cv::Mat& rls_p, std::deque<cv::Mat>& frames, int window_size, cv::Mat& src, cv::Mat& dst);

void rls_adaptative_window(cv::Mat& rls_w, cv::Mat& rls_p, std::deque<cv::Mat>& frames, int window_size, cv::Mat& src, cv::Mat& dst);
} /* namespace denoising */

} /* namespace sonar_processing  */


#endif /* sonar_processing_SonarROI_hpp */
