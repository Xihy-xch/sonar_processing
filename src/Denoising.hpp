#ifndef Denoising_hpp
#define Denoising_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <deque>

namespace sonar_processing {

namespace denoising {

// Recursive Least Squares Filter
class RLS {

public:
    RLS()
        : rls_w()
        , rls_p()
        , frames()
        , window_size(8)
        , mse_0(0)
        {};

    RLS(unsigned int _window_size)
        : rls_w()
        , rls_p()
        , frames()
        , window_size(_window_size)
        , mse_0(0)
        {};

    ~RLS(){};

    void infinite_window(cv::InputArray _src, cv::OutputArray _dst);
    void sliding_window(cv::InputArray _src, cv::OutputArray _dst);
    void adaptative_window(cv::InputArray _src, cv::OutputArray _dst);
    void setWindow_size(uint value) { window_size = value; };
    uint getWindow_size() { return window_size; };

protected:
    cv::Mat rls_w, rls_p;
    std::deque<cv::Mat> frames;
    uint window_size;
    double mse_0;
};

void homomorphic_filter(cv::InputArray _src, cv::OutputArray _dst, int iterations);

} /* namespace denoising */

} /* namespace sonar_processing */

#endif /* sonar_processing_SonarROI_hpp */
