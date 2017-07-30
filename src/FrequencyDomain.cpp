#include "FrequencyDomain.hpp"

namespace sonar_processing {

namespace frequency_domain {

void filters::ideal_lowpass(const cv::Size& size, double cutoff, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    double cx = w / 2;
    double cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = (y - cy) / cy;
            double dx = (x - cx) / cx;
            double r = sqrt(dx * dx + dy * dy);
            if (r < cutoff) {
                filter.at<float>(y, x) = 1.0;
            }
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::butterworth_lowpass(const cv::Size& size, double D, int n, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2.0;
    int cy = h / 2.0;

    cv::Mat filter = cv::Mat::zeros(size, CV_8UC1);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double r = sqrt(dx * dx + dy * dy);
            filter.at<uchar>(y, x) = (1.0 / (1.0 + pow(r / D , 2.0 * n))) * 255;
        }
    }

    filter.convertTo(filter, CV_32F, 1.0/255.0);

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::gaussian_lowpass(const cv::Size& size, double sigma, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2;
    int cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double radius = sqrt(dx * dx + dy * dy);
            filter.at<float>(y, x) = exp(-pow(radius, 2) / (2 * pow(sigma, 2)));
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::ideal_highpass(const cv::Size& size, double cutoff, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    double cx = w / 2;
    double cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = (y - cy) / cy;
            double dx = (x - cx) / cx;

            double r = sqrt(dx * dx + dy * dy);
            if (r > cutoff) {
                filter.at<float>(y, x) = 1.0;
            }
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::butterworth_highpass(const cv::Size& size, double D, int n, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    double cx = w / 2.0;
    double cy = h / 2.0;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double r = sqrt(dx * dx + dy * dy);
            filter.at<float>(y, x) = 1.0 / (1.0 + pow(D / r, 2.0 * n));
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::gaussian_highpass(const cv::Size& size, double sigma, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2;
    int cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double radius = sqrt(dx * dx + dy * dy);
            filter.at<float>(y, x) = 1 - exp(-pow(radius, 2) / (2 * pow(sigma, 2)));
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::ideal_bandreject(const cv::Size& size, double D, double W, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    double cx = w / 2;
    double cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = (y - cy) / cy;
            double dx = (x - cx) / cx;
            double radius = sqrt(dx * dx + dy * dy);
            if (radius < D - W / 2.0 || radius > D + W / 2.0) {
                filter.at<float>(y, x) = 1.0;
            }
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::butterworth_bandreject(const cv::Size& size, double D, int n, double W, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2;
    int cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double r = sqrt(dx * dx + dy * dy);
            double den = r * W;
            double num = r * r - D * D;
            filter.at<float>(y, x) = 1.0 / (1.0 + pow(den / num, 2.0 * n));
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::gaussian_bandreject(const cv::Size& size, double sigma, double W, cv::OutputArray dst) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2;
    int cy = h / 2;

    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;
            double radius = sqrt(dx * dx + dy * dy);
            double den = radius * radius - sigma * sigma;
            double num = radius * W;
            filter.at<float>(y, x) = 1 - exp(-0.5 * pow(den / num, 2));
        }
    }

    cv::Mat to_merge[] = {filter, filter};
    cv::merge(to_merge, 2, dst);
}

void filters::central_frequencies_reject(const cv::Size& size, double D, int n, int center_distance, cv::OutputArray dstx, cv::OutputArray dsty) {
    uint32_t w = size.width;
    uint32_t h = size.height;

    int cx = w / 2;
    int cy = h / 2;

    cv::Mat filterx = cv::Mat(size, CV_8UC1);
    cv::Mat filtery = cv::Mat(size, CV_8UC1);

    filterx.setTo(255);
    filtery.setTo(255);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            double dy = y - cy;
            double dx = x - cx;

            double rx = sqrt(dx * dx);
            double ry = sqrt(dy * dy);
            double r = sqrt(dx * dx + dy * dy);

            if (r > center_distance) {
                filterx.at<uchar>(y, x) = 1.0 / (1.0 + pow(D / rx, 2.0 * n)) * 255;
                filtery.at<uchar>(y, x) = 1.0 / (1.0 + pow(D / ry, 2.0 * n)) * 255;
            }
        }
    }

    filterx.convertTo(filterx, CV_32F, 1.0/255.0);
    filtery.convertTo(filtery, CV_32F, 1.0/255.0);

    cv::Mat to_mergex[] = {filterx, filterx};
    cv::merge(to_mergex, 2, dstx);

    cv::Mat to_mergey[] = {filtery, filtery};
    cv::merge(to_mergey, 2, dsty);
}

void filters::noise_removal(cv::InputArray src_arr, cv::OutputArray dst_arr, double D0, int n0, double D1, int n1, int center_distance) {
    cv::Mat src = src_arr.getMat();
    cv::Mat freq;
    dft::forward(src, freq);
    dft::shift(freq);

    cv::Mat tf_x, tf_y;
    cv::Mat tf_bwlp;

    filters::butterworth_lowpass(freq.size(), D0, n0, tf_bwlp);
    filters::central_frequencies_reject(freq.size(), D1, n1, center_distance, tf_x, tf_y);

    freq = tf_bwlp.mul(freq);
    freq = tf_x.mul(freq);
    freq = tf_y.mul(freq);

    cv::Mat mat;
    dft::inverse_abs(freq, mat);
    cv::normalize(mat, mat, 0, 1, cv::NORM_MINMAX);
    mat(cv::Rect(0, 0, src.cols, src.rows)).copyTo(dst_arr);
}

void dft::shift(cv::Mat src) {
    int cx = src.cols/2;
    int cy = src.rows/2;

    cv::Mat q0(src, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(src, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(src, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(src, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

}

void dft::forward(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    cv::Size size(cv::getOptimalDFTSize(src.cols),
                  cv::getOptimalDFTSize(src.rows));

    cv::Mat padded;
    cv::copyMakeBorder(src, padded,
                       0, size.height - src.rows,
                       0, size.width - src.cols,
                       cv::BORDER_CONSTANT,
                       cv::Scalar::all(0));

    cv::Mat planes[] = {
        cv::Mat_<float>(padded),
        cv::Mat::zeros(padded.size(), CV_32F)
    };

    cv::Mat complex;
    cv::merge(planes, 2, complex);
    cv::dft(complex, complex);
    complex.copyTo(dst_arr);
}

void dft::inverse(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::idft(src_arr.getMat(), dst_arr, cv::DFT_SCALE);
}

void dft::abs(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();
    cv::Mat mag;
    cv::Mat planes[2];
    cv::split(src, planes);
    cv::magnitude(planes[0], planes[1], mag);
    mag += cv::Scalar(1);
    cv::log(mag, mag);
    mag.copyTo(dst_arr);
}

void dft::inverse_abs(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat im;
    dft::inverse(src_arr.getMat(), im);
    dft::abs(im, dst_arr);
}

void dft::show_spectrum(const std::string title, cv::InputArray src_arr) {
    cv::Mat spectrum;
    dft::abs(src_arr.getMat(), spectrum);
    cv::normalize(spectrum, spectrum, 0, 1, cv::NORM_MINMAX);
    cv::imshow(title, spectrum);
}

void dft::show_inverse(const std::string title, cv::InputArray src_arr) {
    cv::Mat im;
    dft::inverse(src_arr.getMat(), im);
    dft::abs(im, im);
    cv::imshow(title, im);
}

} /* namespace frequency_domain */

} /* namespace sonar_processing */
