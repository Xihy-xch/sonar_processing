#include "sonar_target_tracking/ImageUtils.hpp"

namespace sonar_target_tracking {

void image_utils::cv32f_equalize_histogram(cv::Mat src, cv::Mat dst) {
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);
    cv::equalizeHist(aux, aux);
    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

void image_utils::cv32f_clahe(cv::Mat src, cv::Mat dst, double clip_size, cv::Size grid_size) {
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_size, grid_size);
    clahe->apply(aux, aux);

    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

cv::Mat image_utils::zeros_cols(cv::Mat src, std::vector<uint32_t> cols) {
    CV_Assert(src.depth() == CV_32F);

    cv::Mat mat = src;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < cols[y]; x++) {
            mat.at<float>(y, x) = 0;
        }
    }

    return mat;
}

cv::Mat image_utils::horizontal_mirroring(cv::Mat src, std::vector<uint32_t> cols) {

    CV_Assert(src.depth() == CV_32F);

    cv::Mat mat = src;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < cols[y]; x++) {
            mat.at<float>(y, x) = mat.at<float>(y, cols[y] + (cols[y] - x));
        }
    }

    return mat;
}

float image_utils::entropy_8u(const cv::Mat& src, int hist_size) {
    // compute the histogram
    float range[] = {0, 256};
    const float* hist_range = {range};
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;

    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range, uniform, accumulate);
    hist /= src.total();

    // compute the image entropy
    cv::Mat log_p;
    cv::log(hist, log_p);

    float entropy = -1 * cv::sum(hist.mul(log_p)).val[0];
    return entropy;
}

double image_utils::otsu_thresh_8u(const cv::Mat& _src)
{
    cv::Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    const int N = 256;
    int i, j, h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.data + _src.step * i;
        for(j = 0; j < size.width; j++ ) h[src[j]]++;
    }

    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ ) mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON ) continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if( sigma > max_sigma )
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

} /* sonar_target_tracking image_utils */
