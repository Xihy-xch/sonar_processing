#include <cstring>
#include <iostream>
#include <fstream>
#include <iterator>
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/Preprocessing.hpp"

namespace sonar_processing {

cv::Mat image_util::vector32f_to_mat8u(const std::vector<float>& src, int beam_count, int bin_count) {
    cv::Mat dst(beam_count, bin_count, CV_32F, (void*) src.data());
    dst.convertTo(dst, CV_8U, 255);
    return dst;
}

void image_util::equalize_histogram_32f(cv::Mat src, cv::Mat dst) {
    CV_Assert(src.depth() == CV_32F);
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);
    cv::equalizeHist(aux, aux);
    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

void image_util::clahe_32f(cv::Mat src, cv::Mat dst, double clip_size, cv::Size grid_size) {
    CV_Assert(src.depth() == CV_32F);
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);
    clahe_mat8u(aux, aux, clip_size, grid_size);
    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

void image_util::clahe_mat8u(cv::Mat src, cv::Mat& dst, double clip_size, cv::Size grid_size) {
    CV_Assert(src.depth() == CV_8U);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_size, grid_size);
    clahe->apply(src, dst);
}

void image_util::estimate_clahe_parameters(const cv::Mat& src, float& clip_limit, int& grid_size) {
    CV_Assert(src.depth() == CV_8U);
    float best_entropy = 0.0;
    clip_limit = 0.0;
    grid_size = 0;

    for (size_t i = 2; i <= 32; i *= 2) {
        for (float j = 0.02; j <= 4; j += 0.02) {
            cv::Mat dst;
            clahe_mat8u(src, dst, j, cv::Size(i, i));
            float entropy = image_util::entropy(dst, 256);
            if (entropy > best_entropy) {
                best_entropy = entropy;
                clip_limit = j;
                grid_size = i;
            }
        }
    }
}

std::vector<float> image_util::generate_insonification_pattern(const std::vector<std::vector<float> >& frames) {
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

bool image_util::load_insonification_pattern(std::string file_path, std::vector<float>& pattern) {
    std::ifstream input_file(file_path.c_str());
    std::istream_iterator<float> start(input_file), end;
    std::vector<float> sonar_data(start, end);

    if (sonar_data.empty())
        return false;

    pattern = sonar_data;
    return true;
}

void image_util::apply_insonification_correction(std::vector<float>& data, const std::vector<float>& pattern) {
    std::transform(data.begin(), data.end(), pattern.begin(), data.begin(), std::minus<float>());
    std::replace_if(data.begin(), data.end(), std::bind2nd(std::less<float>(), 0), 0);
}

cv::Mat image_util::create_stddev_filter_mask(const cv::Mat& src, uint mask_size) {
    CV_Assert(src.depth() == CV_8U);
    // check if the mask size is valid
    if (mask_size % 2 == 0 || mask_size < 3) {
        std::cout << "Mask size needs to be odd and greater or equal than 3." << std::endl;
        return cv::Mat();
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


cv::Mat image_util::zeros_cols(cv::Mat src, std::vector<uint32_t> cols) {
    CV_Assert(src.depth() == CV_32F);

    cv::Mat mat = src;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < cols[y]; x++) {
            mat.at<float>(y, x) = 0;
        }
    }

    return mat;
}

cv::Mat image_util::horizontal_mirroring(cv::Mat src, std::vector<uint32_t> cols) {

    CV_Assert(src.depth() == CV_32F);

    cv::Mat mat = src;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < cols[y]; x++) {
            mat.at<float>(y, x) = mat.at<float>(y, cols[y] + (cols[y] - x));
        }
    }

    return mat;
}

float image_util::entropy(const cv::Mat& src, int hist_size) {
    // compute the histogram
    cv::Mat hist;
    cv::calcHist(&src, 1, 0, cv::Mat(), hist, 1, &hist_size, 0);
    cv::Mat prob = hist / src.total();

    // compute the image entropy
    float sum = 0;
    for (float* ptr = (float*)prob.datastart; ptr != (float*)prob.dataend; ptr++) {
        float p = *ptr;
        p = (p == 0) ? FLT_EPSILON : p;
        sum += p * log2f(p);
    }

    return -sum;
}

double image_util::otsu_thresh_8u(const cv::Mat& _src, cv::InputArray mask_arr)
{
    cv::Size size = _src.size();
    if( _src.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    const int N = 256;
    int i, j, s, h[N] = {0};
    
    if (mask_arr.empty()) {
        s = size.width*size.height;
        for( i = 0; i < size.height; i++ ) {
            const uchar* src = _src.data + _src.step * i;
            for(j = 0; j < size.width; j++ ) {
                h[src[j]]++;
            }
        }
    }
    else {
        cv::Mat _mask = mask_arr.getMat();
        CV_Assert(_mask.type() == CV_8UC1);
        
        s = 0;

        for( i = 0; i < size.height; i++ ) {

            const uchar* src = _src.data + _src.step * i;
            const uchar* mask = _mask.data + _mask.step * i;

            for(j = 0; j < size.width; j++ ) {
                if (mask[j]) {
                    h[src[j]]++;
                    s++;
                }
            }
        }
    }

    double mu = 0, scale = 1./s;
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

double image_util::otsu_thresh_32f(const cv::Mat& src, cv::InputArray mask_arr) {
    return otsu_thresh_8u(to_mat8u(src, 255.0), mask_arr) / 255.0;
}

cv::Mat image_util::to_mat8u(const cv::Mat& src, double scale) {
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC(src.channels()));
    src.convertTo(dst, CV_8UC(src.channels()), scale);
    return dst;
}

void image_util::adaptative_clahe(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size size, float entropy_thresh, float final_clip_limit) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    float last_entropy = 0;
    for (float clip_limit = FLT_EPSILON; clip_limit < final_clip_limit; clip_limit += 0.2) {
        clahe_mat8u(src, dst, clip_limit, size);

        float current_entropy = entropy(dst);

        if (current_entropy >= entropy_thresh) {
            break;
        }

        if (fabs(last_entropy-current_entropy) < 0.01) {
            break;
        }

        last_entropy = current_entropy;
    }
}

void image_util::copymask(cv::InputArray src_arr, cv::InputArray mask_arr, cv::OutputArray dst_arr) {
    cv::Mat src  = src_arr.getMat();
    cv::Mat mask = mask_arr.getMat();
    uint32_t mask_size = cv::sum(mask)[0] / 255;
    dst_arr.create(cv::Size(1, mask_size), src.type());
    cv::Mat dst = dst_arr.getMat();

    cv::Size sz = src.size();
    size_t esz = dst.elemSize() * dst.channels();

    size_t k = 0;
    dst.setTo(0);

    for(size_t i = 0; i < sz.height; i++) {
        const uchar *mask_ptr = mask.data + mask.step * i;
        const uchar *src_ptr = src.data + src.step * i;

        for(size_t j = 0; j < sz.width; j++) {
            if (mask_ptr[j]) {
                memcpy(dst.data + (k++) * esz, src_ptr + j * esz, esz);
            }
        }
    }
}

void image_util::show_scale(const std::string& title, const cv::Mat& source, double scale) {
    cv::Mat scale_image;
    cv::resize(source, scale_image, cv::Size(source.cols * scale, source.rows * scale));
    cv::imshow(title, scale_image);
}

cv::Rect_<float> image_util::bounding_rect(std::vector<cv::Point2f> points) {
    cv::Point2f top_left = cv::Point2f(FLT_MAX, FLT_MAX);
    cv::Point2f bottom_right = cv::Point2f(FLT_MIN, FLT_MIN);

    for (size_t i = 0; i < points.size(); i++){
        top_left.x = std::min(top_left.x, points[i].x);
        top_left.y = std::min(top_left.y, points[i].y);
        bottom_right.x = std::max(bottom_right.x, points[i].x);
        bottom_right.y = std::max(bottom_right.y, points[i].y);
    }

    return cv::Rect_<float>(top_left, bottom_right);
}

bool image_util::are_equals (const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat diff = image1 != image2;
    return (cv::countNonZero(diff) == 0);
}

void image_util::draw_line(cv::Mat image, std::vector<cv::Point2f>::iterator first, std::vector<cv::Point2f>::iterator last, cv::Scalar line_color) {
    if (first != last) {
        std::vector<cv::Point2f>::iterator it = first;
        while (it != (last-1)) {
            cv::Point2f pt0 = *it;
            cv::Point2f pt1 = *(it+1);
            cv::line(image, pt0, pt1, line_color, 2);
            it++;
        }
    }
}

void image_util::rgb2lab(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    const float t = 0.008856;

    cv::Mat src = src_arr.getMat();

    cv::Mat lab = cv::Mat::zeros(src.size(), CV_32FC3);

    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            float red = src.at<cv::Vec3f>(row, col)[0];
            float green = src.at<cv::Vec3f>(row, col)[1];
            float blue = src.at<cv::Vec3f>(row, col)[2];

            float x = 0.412453 * red + 0.357580 * green + 0.180423 * blue;
            float y = 0.212671 * red + 0.715160 * green + 0.072169 * blue;
            float z = 0.019334 * red + 0.119193 * green + 0.950227 * blue;

            float y3 = pow(y, 1.0/3.0);

            x = x / 0.950456;
            z = z / 1.088754;

            float fx, fy, fz;

            fx = (x > t) ? pow(x, 1.0/3.0) : 7.787 * x + 16.0/116.0;
            fy = (y > t) ? y3 : 7.787 * y + 16.0/116.0;
            fz = (z > t) ? pow(z, 1.0/3.0) : 7.787 * z + 16.0/116.0;

            float l, a, b;
            l = (y > t) ? 116.0 * y3 - 16.0 : 903.3 * y;
            a = 500.0 * (fx - fy);
            b = 200.0 * (fy - fz);

            lab.at<cv::Vec3f>(row, col) = cv::Vec3f(l, a, b);
        }
    }

    lab.copyTo(dst_arr);
}

void image_util::horizontal_normalize(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr, int padding) {
    int w = src_arr.size().width;
    int h = src_arr.size().height;
    int h_offset = h-padding;
    int ksize_h = h_offset/2;
    
    cv::Mat src = src_arr.getMat();
    cv::Mat mask = mask_arr.getMat();
    cv::Mat dst = cv::Mat::zeros(src_arr.size(), src_arr.type());

    for (int y = 0; y < h_offset; y+=ksize_h) {
        int hh = ((y+ksize_h) > (h_offset-1)) ? (h_offset-y-1) : ksize_h;
        cv::Rect rc = cv::Rect(0, y, w, hh);
        cv::normalize(src(rc), dst(rc), 0, 1, cv::NORM_MINMAX, CV_32FC1, mask(rc));
    }
    dst.copyTo(dst_arr);
}

void image_util::draw_contour(cv::InputArray src, cv::OutputArray dst, cv::Scalar color, const std::vector<cv::Point>& contour) {

    if (src.channels() == 1) {
        cv::cvtColor(src, dst, CV_GRAY2BGR);
    }
    else {
        src.getMat().copyTo(dst);
    }

    std::vector<std::vector<cv::Point> > contours;

    contours.push_back(contour);
    cv::drawContours(dst, contours, -1, color, 2);
}

void image_util::draw_mask_contour(cv::InputArray src, cv::OutputArray dst, cv::Scalar color, cv::InputArray mask) {
    std::vector<cv::Point> contour = preprocessing::find_biggest_contour(mask);
    cv::Mat hog_input_canvas;
    draw_contour(src, dst, color, contour);
}

void image_util::draw_rotated_rect(cv::Mat& mat, cv::Scalar color, const cv::RotatedRect& box) {
    cv::Point2f rect_points[4];
    box.points( rect_points );
    for( int i = 0; i < 4; i++ ) {
        cv::line(mat, rect_points[i], rect_points[(i+1)%4], color, 3, 8 );
    }
}

void image_util::rotate(cv::InputArray src_arr, cv::OutputArray dst_arr, float angle, cv::Point2f center, cv::Size dst_size) {
    cv::Mat rot;
    cv::Rect bbox;

    rot = cv::getRotationMatrix2D(center, angle, 1.0);
    bbox = cv::RotatedRect(center, src_arr.size(), angle).boundingRect();

    if (dst_size == cv::Size(-1, -1)) {
        rot.at<double>(0,2) += bbox.width/2.0-center.x;
        rot.at<double>(1,2) += bbox.height/2.0-center.y;
        cv::warpAffine(src_arr, dst_arr, rot, bbox.size());
        return;
    }

    rot.at<double>(0,2) += dst_size.width/2.0-center.x;
    rot.at<double>(1,2) += dst_size.height/2.0-center.y;
    cv::warpAffine(src_arr, dst_arr, rot, dst_size);
}

cv::Rect image_util::get_bounding_rect(cv::InputArray src) {
    std::vector<cv::Point> contour = preprocessing::find_biggest_contour(src);
    return cv::boundingRect(contour);
}

void image_util::find_contour(const cv::Mat& src, std::vector<std::vector<cv::Point> >& contours) {
    cv::Mat im;

    if (src.type() == CV_32F) {
        src.convertTo(im, CV_8U, 255.0);
    }
    else {
        src.copyTo(im);
    }
    cv::findContours(im, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
}

void image_util::draw_contour_min_area_rect(const cv::Mat& src, cv::Mat &dst, std::vector<cv::Point> contour) {
    cv::cvtColor(src, dst, CV_GRAY2BGR);
    if (!contour.empty()) {
        cv::RotatedRect box = cv::minAreaRect(contour);
        draw_rotated_rect(dst, cv::Scalar(0, 255, 0), box);
        cv::circle(dst, box.center, 10, cv::Scalar(0, 255, 0), CV_FILLED);
    }
}

void image_util::create_min_area_rect_mask(const std::vector<cv::Point>& contour, cv::Mat &dst) {
    if (contour.empty()) {
        return;
    }

    create_rotated_rect_mask(cv::minAreaRect(contour), dst);
}

void image_util::create_rotated_rect_mask(const cv::RotatedRect& box, cv::Mat &dst) {
    cv::Point2f rect_points[4];
    box.points(rect_points);

    std::vector<cv::Point> box_contour;
    for (int i = 0; i < 4; i++) {
        box_contour.push_back(rect_points[i]);
    }

    std::vector<std::vector<cv::Point> > box_contours;
    box_contours.push_back(box_contour);
    cv::drawContours(dst, box_contours, -1, cv::Scalar(255), CV_FILLED);
}

void image_util::draw_text(cv::Mat& dst, const std::string& text, cv::Point pos, cv::Scalar color) {
    int baseline = 0;
    double font_scale = 1.5;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 2;
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
    cv::Point textOrg = cv::Point(pos.x, pos.y-text_size.height);
    cv::putText(dst, text, textOrg, font_face, font_scale, color, thickness);
}

} /* sonar_processing image_util */
