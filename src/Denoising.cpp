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
}


/* Recursive Least Square Filter */
void denoising::rls(cv::Mat& rls_w, cv::Mat& rls_p, cv::Mat& src, cv::Mat& dst) {
    // recover data
    if (src.depth() == CV_8U) src.convertTo(src, CV_32F, 1.0 / 255.0);
    src.copyTo(dst);

    if (rls_p.empty() || rls_w.empty()) {
        rls_w = cv::Mat::ones(src.size(), CV_32F) * 0.4;
        rls_p = cv::Mat::ones(src.size(), CV_32F) * 0.75;
    }
    // estimation of w parameters and its covariance p
    else {
        // convert image to linear space
        cv::Mat rls_d;
        cv::log(src, rls_d);

        for (size_t j = 0; j < rls_w.rows; j++) {
            for (size_t i = 0; i < rls_w.cols; i++) {
                float w = rls_w.at<float>(j, i);
                float p = rls_p.at<float>(j, i);
                float d = rls_d.at<float>(j, i);
                p = p - (p * p) / (1 + p);
                w = w + p * (d - w);

                rls_p.at<float>(j, i) = p;
                rls_w.at<float>(j, i) = w;
            }
        }

        // convert back to nonlinear space
        cv::exp(rls_w, dst);
        cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    }
}
        // convert image to logarithmic domain
        cv::Mat rls_d;
        cv::log(src, rls_d);

        for (size_t j = 0; j < rls_p.rows; j++) {
            for (size_t i = 0; i < rls_p.cols; i++) {
                float p = rls_p.at<float>(j, i);
                float w = rls_w.at<float>(j, i);
                float d = rls_d.at<float>(j, i);
                p = p - (p * p) / (1 + p);
                w = w + p * (d - w);

                rls_p.at<float>(j, i) = p;
                rls_w.at<float>(j, i) = w;
            }
        }

        // convert back
        cv::exp(rls_w, dst);
        cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
    }
}
} /* namespace sonar_processing  */
