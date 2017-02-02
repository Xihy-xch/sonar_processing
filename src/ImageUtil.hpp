#ifndef sonar_processing_ImageUtil_hpp
#define sonar_processing_ImageUtil_hpp

#include <iostream>
#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace image_util {

template <typename T>
std::vector<T> mat2vector(cv::Mat mat) {
    std::vector<T> array;
    if (mat.isContinuous()) {
        array.assign((T*)mat.datastart, (T*)mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), (T*)mat.ptr<uchar>(i), (T*)mat.ptr<uchar>(i)+mat.cols);
        }
    }
    return array;
}

template <typename T>
void mat2vector(const cv::Mat& mat, std::vector<T>& array) {
    if (mat.isContinuous()) {
        array.assign((T*)mat.datastart, (T*)mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), (T*)mat.ptr<uchar>(i), (T*)mat.ptr<uchar>(i)+mat.cols);
        }
    }
}

inline void split_channels(const cv::Mat& src, cv::Mat& channel0, cv::Mat& channel1, cv::Mat& channel2) {
    cv::Mat channels[3];
    cv::split(src, channels);
    channel0 = channels[0];
    channel1 = channels[1];
    channel2 = channels[2];
}

template <typename T>
inline T integral_image_sum(const cv::Mat& src, int x1, int y1, int x2, int y2) {
    return src.at<T>(y1, x1)-src.at<T>(y2, x1)+src.at<T>(y2, x2)-src.at<T>(y1, x2);
}

template <typename T>
inline T integral_image_sum(const cv::Mat& src, cv::Rect rc) {
    return integral_image_sum<T>(src, rc.tl().x, rc.tl().y, rc.br().x, rc.br().y);
}

inline void apply_mask(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), src.type());
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    src.copyTo(dst, mask_arr);
    dst.copyTo(dst_arr);
}

inline void show_image(const std::string& title, cv::InputArray src, int scale=1) {

    if (scale == 1) {
        cv::imshow(title, src);
        return;
    }

    cv::Mat scaled; 
    cv::resize(src, scaled, cv::Size(src.size().width/scale, src.size().height/scale));
    cv::imshow(title, scaled);
}

inline void morphology(cv::InputArray src_arr, cv::OutputArray dst_arr, int type, cv::Size kernel_size, int iterations) {
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, kernel_size);
    cv::morphologyEx(src_arr, dst_arr, type, kernel, cv::Point(-1, -1), iterations);
}

inline void erode(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size kernel_size, int iterations) {
    morphology(src_arr, dst_arr, cv::MORPH_ERODE, kernel_size, iterations);
}

inline void dilate(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size kernel_size, int iterations) {
    morphology(src_arr, dst_arr, cv::MORPH_DILATE, kernel_size, iterations);
}

cv::Mat vector32f_to_mat8u(const std::vector<float>& src, int beam_count, int bin_count);

void equalize_histogram_32f(cv::Mat src, cv::Mat dst);

void clahe_32f(cv::Mat src, cv::Mat dst, double clip_size = 40.0, cv::Size grid_size = cv::Size(8, 8));

/**
 * This function minimizes the intensity offsets on the registered areas using the CLAHE technique
 * (Contrast Limited Adaptive Histogram Equalization).
 * @param src: the input image in 8-bit
 * @param dst: the output image in 8-bit
 * @param clip_size: real scalar that specifies a contrast enhancement limit. Higher numbers result in more contrast
 * @param grid_size: size of grid for histogram equalization
 */
void clahe_mat8u(cv::Mat src, cv::Mat& dst, double clip_size = 40.0, cv::Size grid_size = cv::Size(8, 8));

/**
 * This function calculates the clahe parameters based on image entropy
 * @param src: the input image as cv::Mat
 * @param clip_limit: the best clip limit
 * @param grid_size: the best grid size
 */
void estimate_clahe_parameters(const cv::Mat& src, float& clip_limit, int& grid_size);

/**
 * This function calculate the insonification pattern by the mean of N sonar frames
 * @param frames: the array of sonar frames
 * @return the insonification pattern
 */
std::vector<float> generate_insonification_pattern(const std::vector<std::vector<float> >& frames);

/**
 * This function load the insonification pattern from a file
 * @param file_path: the file path name
 * @param pattern: the insonification pattern loaded
 * @return if the insonification pattern was loaded successfully
 */
bool load_insonification_pattern(std::string file_path, std::vector<float>& pattern);

/**
 * This function applies an inhomogeneous insonfication pattern correction on each raw sonar data,
 * originated by different sensitivity of the transducers accross the field of view.
 * @param data: the sonar data to be handled
 * @param pattern: the insonification pattern loaded
 */
void apply_insonification_correction(std::vector<float>& data, const std::vector<float>& pattern);

/**
 * This function create a mask based on standard deviation
 * @param src: the sonar image as cv::Mat
 * @param mask_size: the mask size used (must be odd and equal or greater than 3)
 * @return the mask generated
 */
cv::Mat create_stddev_filter_mask(const cv::Mat& src, uint mask_size);

cv::Mat zeros_cols(cv::Mat src, std::vector<uint32_t> cols);

cv::Mat horizontal_mirroring(cv::Mat src, std::vector<uint32_t> cols);

double otsu_thresh_8u(const cv::Mat& src, cv::InputArray mask_arr = cv::noArray());

double otsu_thresh_32f(const cv::Mat& src, cv::InputArray mask_arr = cv::noArray());

cv::Mat to_mat8u(const cv::Mat& src, double scale);

float entropy(const cv::Mat& src, int hist_size = 256);

void adaptative_clahe(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size size = cv::Size(8, 8), float entropy_thresh = 7.5, float final_clip_limit=2.0);

void copymask(cv::InputArray src_arr, cv::InputArray mask_arr, cv::OutputArray dst_arr);

void show_scale(const std::string& title, const cv::Mat& source, double scale = 1.0);

cv::Rect_<float> bounding_rect(std::vector<cv::Point2f> points);

bool are_equals (const cv::Mat& image1, const cv::Mat& image2);

void draw_line(cv::Mat image, std::vector<cv::Point2f>::iterator first, std::vector<cv::Point2f>::iterator last, cv::Scalar line_color);

void rgb2lab(cv::InputArray src_arr, cv::OutputArray dst_arr);

void horizontal_normalize(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr, int padding=0);

void draw_contour(cv::InputArray src, cv::OutputArray dst, cv::Scalar color, const std::vector<cv::Point>& contour);

void draw_mask_contour(cv::InputArray src, cv::OutputArray dst, cv::Scalar color, cv::InputArray mask);

void draw_rotated_rect(cv::Mat& mat, cv::Scalar color, const cv::RotatedRect& box);

void rotate(cv::InputArray src_arr, cv::OutputArray dst_arr, float angle, cv::Point2f center, cv::Size dst_size = cv::Size(-1, -1));

cv::Rect get_bounding_rect(cv::InputArray src);

void find_contour(const cv::Mat& src, std::vector<std::vector<cv::Point> >& contours);

void draw_contour_min_area_rect(const cv::Mat& src, cv::Mat &dst, std::vector<cv::Point> contour);

void create_min_area_rect_mask(const std::vector<cv::Point>& contour, cv::Mat &dst);

void create_rotated_rect_mask(const cv::RotatedRect& box, cv::Mat &dst);

void draw_text(cv::Mat& dst, const std::string& text, cv::Point pos, cv::Scalar color);

} /* namespace image_util */

} /* sonar_processing image_util */

#endif /* ImageUtils_hpp */
