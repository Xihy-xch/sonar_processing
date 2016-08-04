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

void image_utils::clahe_mat8u(cv::Mat src, cv::Mat& dst, double clip_size, cv::Size grid_size) {
    CV_Assert(src.depth() == CV_8U);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_size, grid_size);
    clahe->apply(src, dst);
}

void image_utils::estimate_clahe_parameters(const cv::Mat& src, float& clip_limit, int& grid_size) {
    CV_Assert(src.depth() == CV_8U);
    float best_entropy = 0.0;
    clip_limit = 0.0;
    grid_size = 0;

    for (size_t i = 2; i <= 32; i *= 2) {
        for (float j = 0.02; j <= 4; j += 0.02) {
            cv::Mat dst;
            clahe_mat8u(src, dst, j, cv::Size(i, i));
            float entropy = entropy_8u(dst);
            if (entropy > best_entropy) {
                best_entropy = entropy;
                clip_limit = j;
                grid_size = i;
            }
        }
    }
}

std::vector<float> image_utils::generate_insonification_pattern(const std::vector<std::vector<float> >& frames) {
    std::vector<float> pattern;
    pattern.clear();

    for (unsigned int i = 0; i < frames[0].size(); ++i) {
        float accum = 0.0;

        for (unsigned int j = 0; j < frames.size(); ++j)
            accum += frames[j][i];

        accum /= frames.size();
        pattern.push_back(accum);
    }

    return pattern;
}

bool image_utils::load_insonification_pattern(std::string file_path, std::vector<float>& pattern) {
    std::istream_iterator<float> start(input_file), end;
    std::ifstream input_file(file_path.c_str());
    std::vector<float> sonar_data(start, end);

    if (sonar_data.empty())
        return false;

    pattern = sonar_data;
    return true;
}

void image_utils::apply_insonification_correction(std::vector<float>& data, const std::vector<float>& pattern) {
    std::transform(data.begin(), data.end(), pattern.begin(), data.begin(), std::minus<float>());
    std::replace_if(data.begin(), data.end(), std::bind2nd(std::less<float>(), 0), 0);
}

cv::Mat image_utils::create_stddev_filter_mask(const cv::Mat& src, uint mask_size) {
    CV_Assert(src.depth() == CV_8U);
    // check if the mask size is valid
    if (mask_size % 2 == 0 || mask_size < 3) {
        std::cout << "Mask size needs to be odd and greater or equal than 3." << std::endl;
        return 0;
    }

    // create the stddev filter
    cv::Mat mat;
    src.convertTo(mat, CV_32F, 1.0 / 255);

    cv::Mat dst = cv::Mat(mat.size(), mat.type());
    cv::Mat mask = cv::Mat::ones(mask_size, mask_size, CV_32F);
    int n = cv::sum(mask).val[0];
    int n1 = n - 1;

    if (n != 1) {
        cv::Mat conv1, conv2;
        cv::filter2D(mat.mul(mat), conv1, -1, mask / n1, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        cv::filter2D(mat, conv2, -1, mask, cv::Point(-1, -1), 0, cv::BORDER_REFLECT);
        conv2 = conv2.mul(conv2) / (n * n1);
        sqrt(cv::max((conv1-conv2), 0), dst);
    }
    else
        dst = cv::Mat(mat.size(), mat.type());

    return dst;
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
