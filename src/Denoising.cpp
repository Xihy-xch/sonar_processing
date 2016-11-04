#include "base/Plot.hpp"
#include "sonar_processing/Denoising.hpp"

namespace sonar_processing {

// Homomorphic Filtering with the log function
void denoising::homomorphic_filter(cv::InputArray _src, cv::OutputArray _dst, int iterations) {
    cv::Mat src = _src.getMat();
    cv::Mat mat;
    src.copyTo(mat);

    // check image depth
    if (mat.depth() == CV_8U)
        mat.convertTo(mat, CV_32F, 1.0 / 255.0);

    // apply n iterations of the algoritm
    mat.copyTo(_dst);
    cv::Mat dst = _dst.getMat();

    for (size_t i = 0; i < iterations; i++) {
        if (i > 1) dst.copyTo(mat);
        cv::Mat result;
        cv::log(mat, result);
        cv::medianBlur(result, result, 5);
        cv::exp(result, result);
        result.copyTo(dst);
    }

    dst.convertTo(dst, CV_8UC1, 255);
} /* namespace roi */

} /* namespace sonar_processing  */
