#ifndef QualityMetrics_hpp
#define  QualityMetrics_hpp

#include <iostream>
#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {

namespace qs {
    /**
     * Mean Square Error (MSE).
     * Measures the quality change between the original image and the de-speckled image.
     * @param I1: original image
     * @param I2: denoised image
     * @return the MSE value.
     */
    double MSE(cv::Mat& I1, cv::Mat& I2);

    /**
     * Root Mean Square Error.
     * Measure the square root of the squared error averaged over a pixel window. It is
     * the best approximation of the standard error.
     * @param I1: original image
     * @param I2: denoised image
     * @return the MSE value.
     */
    double RMSE(cv::Mat& I1, cv::Mat& I2);

    /**
     * Peak Signal to Noise Ratio.
     * Provides the quality of the image in terms of power of the original signal and
     * de-noised signal.
     * @param I1: original image
     * @param I2: denoised image
     * @return the MSE value.
     */
    double PSNR(cv::Mat& I1, cv::Mat& I2);

    /**
     * Multiscale Structural Similarity.
     * @param I1: original image
     * @param I2: denoised image
     * @return the MSSIM value.
     */
    cv::Scalar MSSIM(const cv::Mat& I1, const cv::Mat& I2);
}
}

#endif
