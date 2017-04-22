#include "base/Plot.hpp"
#include "sonar_processing/Denoising.hpp"
#include "sonar_processing/QualityMetrics.hpp"

namespace sonar_processing {

namespace denoising {

// Standard Recursive Least Square Filter algorithm
cv::Mat RLS::infinite_window(const cv::Mat& src) {
    CV_Assert(src.depth() == CV_32F);
    cv::Mat dst;

    // initialize coefficients
    if (rls_p.empty() || rls_w.empty() || src.size() != rls_p.size()) {
        rls_p = cv::Mat::ones(src.size(), CV_32F);
        src.copyTo(rls_w);
        src.copyTo(dst);
        frames.clear();
    }

    // estimation of w parameters and its covariance p
    else {
        // convert image to linear space
        cv::Mat rls_d;
        cv::log(src, rls_d);

        for (size_t i = 0; i < rls_w.rows; i++) {
            for (size_t j = 0; j < rls_w.cols; j++) {
                float p = rls_p.at<float>(i, j);
                float w = rls_w.at<float>(i, j);
                float d = rls_d.at<float>(i, j);
                p = p - ((p * p) / (1 + p));
                w = w + p * (d - w);

                rls_p.at<float>(i, j) = p;
                rls_w.at<float>(i, j) = w;
            }
        }

        // convert back to nonlinear space
        cv::exp(rls_w, dst);
    }
    return dst;
}

// Recursive Least Square Filter algorithm with fixed data window
cv::Mat RLS::sliding_window(const cv::Mat& src) {
    CV_Assert(src.depth() == CV_32F);

    // check for downdate
    if (frames.size() == window_size) {
        cv::Mat oldest;
        cv::log(frames.front(), oldest);
        frames.pop_front();

        // remove old measurement
        for (size_t i = 0; i < rls_w.rows; i++) {
            for (size_t j = 0; j < rls_w.cols; j++) {
                float p = rls_p.at<float>(i, j);
                float w = rls_w.at<float>(i, j);
                float oldest_d = oldest.at<float>(i, j);
                p = p + ((p * p) / (1 - p));
                w = w - p * (oldest_d - w);

                rls_p.at<float>(i, j) = p;
                rls_w.at<float>(i, j) = w;
            }
        }
    }

    // store current frame
    frames.push_back(src);

    // update coefficients
    return infinite_window(src);
}

// Recursive Least Square Filter algorithm with dynamic data window size
cv::Mat RLS::adaptive_window(const cv::Mat& src) {
    CV_Assert(src.depth() == CV_32F);

    // if window size is higher than frame size, decrease 1x
    if (frames.size() > window_size) {
        cv::Mat oldest;
        cv::log(frames.front(), oldest);
        frames.pop_front();

        // remove old measurement
        for (size_t i = 0; i < rls_w.rows; i++) {
            for (size_t j = 0; j < rls_w.cols; j++) {
                float p = rls_p.at<float>(i, j);
                float w = rls_w.at<float>(i, j);
                float oldest_d = oldest.at<float>(i, j);
                p = p + ((p * p) / (1 - p));
                w = w - p * (oldest_d - w);

                rls_p.at<float>(i, j) = p;
                rls_w.at<float>(i, j) = w;
            }
        }
    }

    // run sliding window and update coefficients
    cv::Mat dst = sliding_window(src);

    // adaptative window size
    double mse_i = qs::MSE(src, dst);
    if (!mse_0 && mse_i && frames.size() == window_size) mse_0 = mse_i;
    if (mse_0) {
        if ((mse_i <= mse_0) && (window_size < 10)) window_size++;
        if ((mse_i > mse_0) && (window_size > 2)) window_size--;
    }
    return dst;
}

// Homomorphic Filtering with the log function
void homomorphic_filter(cv::InputArray _src, cv::OutputArray _dst, int iterations) {
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
}

} /* namespace denoising */

} /* namespace sonar_processing */
