#ifndef sonar_processing_FrequencyDomain_hpp
#define sonar_processing_FrequencyDomain_hpp

#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace  frequency_domain {

namespace filters {

    void ideal_lowpass(const cv::Size& size, double cutoff, cv::OutputArray dst);
    void butterworth_lowpass(const cv::Size& size, double D, int n, cv::OutputArray dst);
    void gaussian_lowpass(const cv::Size& size, double sigma, cv::OutputArray dst);

    void ideal_highpass(const cv::Size& size, double cutoff, cv::OutputArray dst);
    void butterworth_highpass(const cv::Size& size, double D, int n, cv::OutputArray dst);
    void gaussian_highpass(const cv::Size& size, double sigma, cv::OutputArray dst);

    void ideal_bandreject(const cv::Size& size, double D, double W, cv::OutputArray dst);
    void butterworth_bandreject(const cv::Size& size, double D, int n, double W, cv::OutputArray dst);
    void gaussian_bandreject(const cv::Size& size, double sigma, double W, cv::OutputArray dst);

} /* namespace filters */

namespace dft {

    void shift(cv::Mat src);
    void forward(cv::InputArray src_arr, cv::OutputArray dst_arr);
    void abs(cv::InputArray src_arr, cv::OutputArray dst_arr);
    void inverse(cv::InputArray src_arr, cv::OutputArray dst_arr);
    void show_spectrum(const std::string title, cv::InputArray src_arr);
    void show_inverse(const std::string title, cv::InputArray src_arr);
    void inverse_abs(cv::InputArray src_arr, cv::OutputArray dst_arr);

} /* namespace dft */

} /* namespace frequency_domain */

} /* namespace sonar_processing */

#endif /* end of include guard: sonar_processing_FrequencyDomain_hpp */
