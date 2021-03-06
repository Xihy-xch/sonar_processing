#include "Preprocessing.hpp"
#include "ImageUtil.hpp"
#include "Utils.hpp"


// C++ includes
#include <numeric>
#include <algorithm>

namespace sonar_processing {

cv::Rect preprocessing::calc_horiz_roi_old(cv::Mat src) {
    cv::Mat col_sum;

    cv::reduce(src, col_sum, 0, CV_REDUCE_SUM);
    cv::blur(col_sum, col_sum, cv::Size(25, 25));

    double min, max;
    cv::minMaxLoc(col_sum, &min, &max);
    float thresh = ((min + max) / 2.0) * 0.5;
    cv::Mat bin;
    cv::threshold(col_sum, bin, thresh, 255, cv::THRESH_BINARY);
    bin.convertTo(bin, CV_8UC1, 1);

    cv::Mat bin_area;
    cv::repeat(bin, src.cols, 1, bin_area);

    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(bin_area, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

    float min_x = src.cols * 0.05;
    float min_width = src.cols * 0.5;

    std::vector<uint32_t> left_values;
    std::vector<uint32_t> right_values;

    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        if ( rc.x > min_x && rc.width > min_width ) {
            left_values.push_back(rc.tl().x);
            right_values.push_back(rc.br().x);
        }
    }

    cv::Rect rc(cv::Point(0, 0), src.size());
    if (!left_values.empty() && !right_values.empty()) {
        uint32_t l = *std::min_element(left_values.begin(), left_values.end());
        uint32_t r = *std::max_element(right_values.begin(), right_values.end());
        rc.x = l;
        rc.width = r - l;
    }

    return rc;
}

cv::Rect preprocessing::calc_horiz_roi(cv::Mat src, float alpha) {
    cv::Mat col_sum;

    cv::reduce(src, col_sum, 0, CV_REDUCE_SUM);
    cv::blur(col_sum, col_sum, cv::Size(25, 25));

    cv::Mat acc_sum = cv::Mat::zeros(col_sum.size(), col_sum.type());
    acc_sum.at<float>(0, 0) = col_sum.at<float>(0, 0);
    for (int i = 1; i < col_sum.cols; i++) {
        acc_sum.at<float>(0, i) = acc_sum.at<float>(0, i-1) + col_sum.at<float>(0, i);
    }

    double min, max;
    cv::minMaxLoc(acc_sum, &min, &max);

    float thresh = alpha * (max - min) + min;
    cv::Mat bin;
    cv::threshold(acc_sum, bin, thresh, 1.0, cv::THRESH_BINARY);

    std::vector<float> bin_vec = image_util::mat2vector<float>(bin);
    uint32_t new_x = std::find(bin_vec.begin(), bin_vec.end(), 1) - bin_vec.begin();

    return cv::Rect(cv::Point(new_x, 0), cv::Size(acc_sum.cols - new_x, src.rows));
}

void preprocessing::adaptive_clahe(cv::InputArray _src, cv::OutputArray _dst, unsigned int max_clip_limit) {
    cv::Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    cv::Mat dst = _dst.getMat();

    float best_entropy = 0.0;
    for (float c = 0.25; c <= max_clip_limit; c++) {
        cv::Mat enhanced;
        image_util::clahe_mat8u(src, enhanced, c, cv::Size(4, 4));
        float entropy = image_util::entropy(enhanced, 256);
        if (entropy > best_entropy) {
            best_entropy = entropy;
            enhanced.copyTo(dst);
        }
    }
}

double preprocessing::horiz_difference(cv::Mat src) {
    CV_Assert(src.depth() == CV_32F || src.depth() == CV_8U);

    cv::Mat mat;

    if (src.depth() == CV_8U) {
        src.convertTo(mat, CV_32F);
    }
    else {
        mat = src;
    }

    float dsum = 0;
    for (int y = 0; y < src.rows-1; y++) {
        for (int x = 0; x < src.cols-1; x++) {
            float dd = mat.at<float>(y, x) - mat.at<float>(y,x+1);
            dsum += abs(dd);
        }
    }
    return dsum / src.total();
}

double preprocessing::vert_difference(cv::Mat src) {
    CV_Assert(src.depth() == CV_32F || src.depth() == CV_8U);

    cv::Mat mat;

    if (src.depth() == CV_8U) {
        src.convertTo(mat, CV_32F);
    }
    else {
        mat = src;
    }

    float dsum = 0;
    for (int x = 0; x < src.cols-1; x++) {
        for (int y = 0; y < src.rows-1; y++) {
            dsum += abs(mat.at<float>(y, x) - mat.at<float>(y+1,x));
        }
    }
    return dsum / src.total();
}

std::vector<std::vector<cv::Point> > preprocessing::convexhull(std::vector<std::vector<cv::Point> > contours) {
        std::vector<std::vector<cv::Point> >hull(contours.size());

        for( uint32_t i = 0; i < contours.size(); i++ ) {
            cv::convexHull( cv::Mat(contours[i]), hull[i], false );
        }

        return hull;
}

std::vector<std::vector<cv::Point> > preprocessing::find_contours(cv::Mat src, int mode, bool convex_hull) {
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat im_contours;
    src.copyTo(im_contours);

    cv::findContours(im_contours, contours, mode, CV_CHAIN_APPROX_SIMPLE);

    if (convex_hull) {
        return convexhull(contours);
    }

    return contours;

}

std::vector<std::vector<cv::Point> > preprocessing::adaptative_find_contours_and_filter(cv::Mat src, double area_factor, double width_factor, double height_factor) {
    std::vector<std::vector<cv::Point> > filtering_contours, nonzero_area_contours;
    std::vector<std::vector<cv::Point> > contours = find_contours(src);

    std::vector<double> height_vals;
    std::vector<double> width_vals;
    std::vector<double> area_vals;

    for( int i = 0; i < contours.size(); i++ ) {
        double area = cv::contourArea(contours[i]);
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );

        if (area > 0) {
            area_vals.push_back(area);
            height_vals.push_back(rc.height);
            width_vals.push_back(rc.width);
            nonzero_area_contours.push_back(contours[i]);
        }
    }

    double area_max = *std::max_element(area_vals.begin(), area_vals.end());
    double area_min = *std::min_element(area_vals.begin(), area_vals.end());
    double area_thresh = ((area_max + area_min) / 2.0) * area_factor;

    double width_max = *std::max_element(width_vals.begin(), width_vals.end());
    double width_min = *std::min_element(width_vals.begin(), width_vals.end());
    double width_thresh = ((width_max + width_min) / 2.0) * width_factor;

    double height_max = *std::max_element(height_vals.begin(), height_vals.end());
    double height_min = *std::min_element(height_vals.begin(), height_vals.end());
    double height_thresh = ((height_max + height_min) / 2.0) * height_factor;

    for( int i = 0; i < nonzero_area_contours.size(); i++ ) {
        double area = cv::contourArea(nonzero_area_contours[i]);
        cv::Rect rc = cv::boundingRect( cv::Mat(nonzero_area_contours[i]) );

        if (area > area_thresh && rc.width > width_thresh && rc.height > height_thresh){
            filtering_contours.push_back(nonzero_area_contours[i]);
        }
    }

    return filtering_contours;
}

std::vector<std::vector<cv::Point> > preprocessing::find_contours_and_filter(cv::Mat src, cv::Size min_size, int mode, bool convex_hull) {
    std::vector<std::vector<cv::Point> > filtering_contours;
    std::vector<std::vector<cv::Point> > contours = find_contours(src, mode, convex_hull);

    int min_area = min_size.width * min_size.height;

    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        if (rc.width > min_size.width &&
            rc.height > min_size.height) {
            filtering_contours.push_back(contours[i]);
        }
    }

    return filtering_contours;
}

void preprocessing::mean_horiz_difference_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize,
                                               double mean_thresh, double horiz_difference_thresh) {

    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    for (int y = 0; y < src.rows-bsize; y+=bsize/4) {
        for (int x = 0; x < src.cols-bsize; x+=bsize/4) {
            cv::Rect roi = cv::Rect(x, y, bsize, bsize);
            cv::Mat block;
            src(roi).copyTo(block);

            cv::Scalar sum = cv::sum(block);
            double m = sum[0] / (bsize * bsize);
            double d = horiz_difference(block);

            if (m > mean_thresh && d < horiz_difference_thresh) {
                dst(cv::Rect(x, y, bsize/4, bsize/4)).setTo(1.0);
            }
        }
    }
}

void preprocessing::mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    for (int y = 0; y < src.rows-bsize; y+=bsize/4) {
        for (int x = 0; x < src.cols-bsize; x+=bsize/4) {
            cv::Rect roi = cv::Rect(x, y, bsize, bsize);
            cv::Mat block;
            src(roi).copyTo(block);
            cv::Scalar mean = cv::mean(block);
            cv::Mat means = cv::Mat::ones(cv::Size(bsize, bsize), block.type()) * mean[0];
            dst(cv::Rect(x, y, bsize, bsize)) += means;
        }
    }
}

void preprocessing::difference_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    for (int y = 0; y < src.rows-bsize; y++) {
        for (int x = 0; x < src.cols-bsize; x++) {
            cv::Rect roi = cv::Rect(x, y, bsize, bsize);
            cv::Mat block;
            src(roi).copyTo(block);
            double x_diff = horiz_difference(block);
            double y_diff = vert_difference(block);
            dst.at<float>(y, x) += sqrt(x_diff * x_diff + y_diff * y_diff);
        }
    }
}

uint32_t preprocessing::find_first_higher(cv::Mat mat, uint32_t row) {
    cv::Mat line;
    mat.row(row).copyTo(line);
    cv::blur(line, line, cv::Size(25, 25));

    double min, max;
    cv::minMaxLoc(line, &min, &max);

    double thresh = ((min + max) / 2.0) * 0.3;

    for (int col = 0; col < mat.cols; col++) {
        if (line.at<float>(0, col) > thresh) return col;
    }
}

void preprocessing::contrast_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int div_size) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    src.copyTo(dst);

    const double max_entropy_thesh = 7.5;
    const uint32_t bsize = src.cols/div_size;

    for (int x = 0, i = 0; x < src.cols; x+=bsize, i++) {
        uint32_t r = (src.cols > x + bsize) ? bsize : src.cols - x;
        cv::Rect roi = cv::Rect(x, 0, r, src.rows);
        cv::Mat block = dst(roi);
        image_util::adaptative_clahe(block, block, cv::Size(8, 8), max_entropy_thesh);
    }
}

void preprocessing::gradient_filter(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    cv::Mat Gx, Gy;
    cv::Mat Gx2, Gy2;
    cv::Sobel( src, Gx, CV_16S, 1, 0, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gx, Gx2 );

    cv::Sobel( src, Gy, CV_16S, 0, 1, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gy, Gy2 );

    cv::addWeighted(Gx2, 0.5, Gy2, 0.5, 0, dst);
}

void preprocessing::weak_target_thresholding_old(cv::InputArray src_arr, cv::InputArray src_hc_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();
    cv::Mat src_hc = src_hc_arr.getMat();

    dst_arr.create(src.size(), CV_8UC1);
    cv::Mat dst = dst_arr.getMat();
    dst.setTo(0);

    const uint32_t max_win_width = 64;
    const uint32_t max_win_height = 64;
    const double high_intensities_thesh = 190;
    const double high_intensities_mean_thresh = 0.01;
    const double gradient_thresh = 85.0;
    const double difference_thresh = 50.0;

    uint32_t src_width = src.cols;
    uint32_t src_height = src.rows;

    cv::Mat high_bin;
    cv::threshold(src_hc, high_bin, high_intensities_thesh, 1, cv::THRESH_BINARY);

    for (int y = 0; y < src_height; y+=max_win_height/2) {
        for (int x = 0; x < src_width; x+=max_win_width/2) {

            uint32_t win_width = (src_width > x + max_win_width) ? max_win_width : src_width - x;
            uint32_t win_height = (src_height > y + max_win_height) ? max_win_height : src_height - y;

            if (win_width < max_win_width / 2) continue;
            if (win_height < max_win_height / 2) continue;

            cv::Rect win_rect = cv::Rect(x, y, win_width, win_height);

            if ((cv::mean(high_bin(win_rect))[0]) > high_intensities_mean_thresh) {
                cv::Mat win;
                cv::blur(src(win_rect), win, cv::Size(15, 15));

                cv::Mat grad, grad_bin;
                gradient_filter(win, grad);

                cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX);
                cv::threshold(grad, grad_bin, gradient_thresh, 255, cv::THRESH_BINARY);

                double vert_diff = vert_difference(grad_bin);
                double horiz_diff = horiz_difference(grad_bin);
                double diff = sqrt(vert_diff * vert_diff + horiz_diff * horiz_diff);

                if (diff < difference_thresh) dst(win_rect) += grad_bin;
            }
        }
    }

    cv::morphologyEx(dst, dst, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5, 5)),
                     cv::Point(-1, -1), 1);
}

float preprocessing::calc_spatial_variation_coefficient(std::vector<float> values) {
    cv::Mat mat(values);
    cv::Scalar m, s;
    cv::meanStdDev(mat, m, s);
    float ss = s[0] + 1;
    float mm = m[0];
    float a = pow(ss, 2);
    float b = pow(mm, 2);
    return atanf(mm/ss) * sqrtf(a + b);
}

float preprocessing::spatial_variation_coefficient(cv::Mat src) {

    uint32_t cx = src.cols / 2;
    uint32_t cy = src.rows / 2;

    std::map<int, std::vector<float> > classes;
    for (uint32_t y = 0; y < src.rows; y++){
        for (uint32_t x = 0; x < src.cols; x++) {
            	int dx = x - cx;
            	int dy = y - cy;
            	int d = sqrt(dx * dx + dy * dy);
                if (d != 0) {
                    classes[d].push_back((float)src.at<uchar>(y , x));
                }
        }
    }

    std::vector<float> svc_vals;
    for (std::map<int, std::vector<float> >::iterator it = classes.begin(); it != classes.end(); it++){
        svc_vals.push_back(calc_spatial_variation_coefficient(it->second));
    }

    return calc_spatial_variation_coefficient(svc_vals);
}

void preprocessing::spatial_variation_coefficient_filter(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), CV_32F);
    cv::Mat dst = dst_arr.getMat();

    uint32_t bsize = 16;
    for (int y = 0; y < src.rows-bsize; y+=bsize / 4) {
        for (int x = 0; x < src.cols-bsize; x+=bsize / 4) {
            dst.at<float>(y, x) = spatial_variation_coefficient(src(cv::Rect(x, y, bsize, bsize)));
        }
    }
}

void preprocessing::difference_of_gaussian(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    cv::Mat G0, G1;

    cv::GaussianBlur(src, G0, cv::Size(3, 3), 2.5);
    cv::GaussianBlur(src, G1, cv::Size(5, 5), 2.5);
    cv::absdiff(G0, G1, dst);
}

void preprocessing::simple_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr, double alpha, uint32_t colsdiv, cv::InputArray mask_arr) {

    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), CV_8UC1);
    cv::Mat dst = dst_arr.getMat();

    uint32_t ncols = (uint32_t)ceil(src.cols / (float)colsdiv);

    std::vector<double> max_vals;
    std::vector<double> min_vals;

    std::vector<cv::Rect> rcs;

    for (uint32_t x = 0; x < src.cols; x += ncols) {
        cv::Rect roi = cv::Rect(x, 0, (src.cols >= x + ncols) ? ncols : src.cols - x, src.rows);
        double min, max;

        if (mask_arr.empty()) {
            cv::minMaxLoc(src(roi), &min, &max);
        }
        else {
            cv::Mat mask = mask_arr.getMat();
            cv::minMaxLoc(src(roi), &min, &max, NULL, NULL, mask(roi));
        }
        max_vals.push_back(max);
        min_vals.push_back(min);
        rcs.push_back(roi);
    }

    double max_el = *std::max_element(max_vals.begin(), max_vals.end());

    dst.setTo(0);
    for (int i = 0; i < rcs.size(); i++) {
        double max_p = max_vals[i] / max_el;

        if (max_p > 0.3) {
            double min = min_vals[i];
            double max = max_vals[i];
            double thresh = alpha * (max - min) + min;
            cv::threshold(src(rcs[i]), dst(rcs[i]), thresh, 255, cv::THRESH_BINARY);
        }
    }
}

void preprocessing::houghlines_mask(cv::InputArray src_arr, cv::OutputArray dst_arr, double rho, double theta, int threshold, double min_line_length, double max_line_gap) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), CV_8UC1);
    cv::Mat dst = dst_arr.getMat();

    cv::vector<cv::Vec4i> lines;
    cv::HoughLinesP(src, lines, rho, theta, threshold, min_line_length, max_line_gap);

    dst.setTo(0);
    for( size_t i = 0; i < lines.size(); i++ ){
        cv::Vec4i l = lines[i];
        cv::line( dst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255), 3, CV_AA);
    }
}

std::vector<std::vector<cv::Point> > preprocessing::remove_low_intensities_contours(cv::InputArray src_arr, std::vector<std::vector<cv::Point> > contours) {

    if (contours.size() <= 1) {
        return contours;
    }

    cv::Mat src = src_arr.getMat();

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);

    std::vector<double> high_vals;

    for( int i = 0; i < contours.size(); i++ ) {

        mask.setTo(0);
        cv::drawContours(mask, contours, i, cv::Scalar(255), CV_FILLED);

        double area = cv::contourArea(contours[i]) / src.total();

        if (area > 0.02) {
            cv::Mat block;
            src.copyTo(block, mask);
            cv::normalize(block, block, 1, 255, cv::NORM_MINMAX);

            cv::Mat hist;
            int hist_size = 256;
            float range[] = { 160, 256 } ;
            const float* hist_range = { range };
            cv::calcHist(&block, 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);

            int total_pixels = cv::sum(mask)[0];
            float high_prob = (cv::sum(hist)[0] /  total_pixels) * 10000;

            high_vals.push_back(high_prob);
        }
    }

    double high_min, high_max;
    min_max_element<double>(high_vals, high_min, high_max);

    if ((high_max - high_min) <= 0.01) {
        return contours;
    }

    double high_thresh = high_max * 0.7;

    std::vector<std::vector<cv::Point> > result;

    for( int i = 0; i < contours.size(); i++ ) {
        if (high_vals[i] > high_thresh) {
            result.push_back(contours[i]);
        }
    }

    return result;
}

void preprocessing::remove_blobs(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size min_size, int mode, bool convex_hull) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), CV_8UC1);


    std::vector<std::vector<cv::Point> > contours = find_contours_and_filter(src, min_size, mode, convex_hull);

    if (!contours.empty()) {
        cv::Mat dst = dst_arr.getMat();
        dst.setTo(0);
        cv::drawContours(dst, contours, -1, cv::Scalar(255), CV_FILLED);
    }
}

void preprocessing::weak_target_thresholding(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), CV_8UC1);
    cv::Mat dst = dst_arr.getMat();
    dst.setTo(0);

    cv::Mat grad, src_sm, grad_bin;
    cv::boxFilter(src, src_sm, CV_8U, cv::Size(15, 15));
    gradient_filter(src_sm, grad);
    cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX);
    cv::boxFilter(src_sm, src_sm, CV_8U, cv::Size(50, 50));
    grad-=src_sm;
    cv::normalize(grad, grad, 0, 255, cv::NORM_MINMAX);

    simple_thresholding(grad, grad_bin, 0.2);

    cv::morphologyEx(grad_bin, grad_bin, cv::MORPH_CLOSE,
                 cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5, 5)),
                 cv::Point(-1, -1), 1);

    cv::morphologyEx(grad_bin, grad_bin, cv::MORPH_OPEN,
                 cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5, 5)),
                 cv::Point(-1, -1), 1);



    cv::Mat grad_mask = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int i = 0; i < 2; i++) {

        grad_mask.setTo(0);

        houghlines_mask(grad_bin, grad_bin, 5, CV_PI/180, 10, 10, 40);

        remove_blobs(grad_bin, grad_bin, cv::Size(30, 30), CV_RETR_LIST);

        uint32_t bsize, bstep;

        bsize = 32;
        bstep = bsize / 8;

        for (int y = 0; y < src.rows-bsize; y+=bstep) {
            for (int x = 0; x < src.cols-bsize; x+=bstep) {
                cv::Rect roi = cv::Rect(x, y, bsize, bsize);
                double m = (cv::mean(grad_bin(roi))[0] / 255.0) ;
                if (m > 0.3) grad_mask(roi).setTo(255);
            }
        }

        grad_mask.copyTo(grad_bin);
    }

    std::vector<std::vector<cv::Point> > contours = find_contours_and_filter(grad_mask, cv::Size(80, 80), CV_RETR_EXTERNAL, true);

    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    cv::Mat final_mask = cv::Mat::zeros(src.size(), src.type());

    cv::GaussianBlur(src, src_sm, cv::Size(15, 15), 0);
    gradient_filter(src_sm, grad);

    for (int i = 0; i < contours.size(); i++) {

        mask.setTo(0);
        cv::drawContours(mask, contours, i, cv::Scalar(255), CV_FILLED);
        grad_bin.setTo(0);
        grad.copyTo(grad_bin, mask);

        cv::Mat bin;
        double min, max;
        cv::minMaxLoc(grad_bin, &min, &max, NULL, NULL, mask);
        double delta = (max - min) * 0.1;
        double thresh = image_util::otsu_thresh_8u(grad_bin);

        cv::threshold(grad_bin, bin, thresh + delta, 255, cv::THRESH_BINARY);
        remove_blobs(bin, grad_bin, cv::Size(5, 5), CV_RETR_LIST);

        std::vector<std::vector<cv::Point> > blobs_contours = find_contours(grad_bin, CV_RETR_LIST);

        std::vector<double> height_vals;
        std::vector<double> width_vals;

        for (int j = 0; j < blobs_contours.size(); j++) {
            cv::Rect blob_rc = cv::boundingRect(blobs_contours[j]);
            height_vals.push_back(blob_rc.height);
            width_vals.push_back(blob_rc.width);
        }

        double height_thresh = min_max_thresh<double>(height_vals, 0.1);
        double width_thresh  = min_max_thresh<double>(width_vals, 0.1);

        grad_bin.setTo(0);
        for (int j = 0; j < blobs_contours.size(); j++) {
            if (height_vals[j] > height_thresh && width_vals[j] > width_thresh) {
                cv::drawContours(grad_bin, blobs_contours, j, cv::Scalar(255), CV_FILLED);
            }
        }

        cv::morphologyEx(grad_bin, grad_bin, cv::MORPH_DILATE,
            cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5, 5)),
            cv::Point(-1, -1), 1);

        houghlines_mask(grad_bin, mask, 5, CV_PI/180, 10, 20, 100);

        std::vector<std::vector<cv::Point> > contours2;
        contours2 = find_contours_and_filter(mask, cv::Size(30, 30), CV_RETR_EXTERNAL, true);
        if (contours2.empty()) continue;

        mask.setTo(0);
        cv::drawContours(mask, contours2, -1, cv::Scalar(255), CV_FILLED);

        final_mask += mask;
    }

    final_mask.copyTo(dst);
}

void preprocessing::remove_low_intensities_columns(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    uint32_t cols_mid = src.cols / 2;
    uint32_t bsize = 16;
    uint32_t bstep = bsize / 4;
    std::vector<float> low_intensities_prob;

    for (int x = cols_mid; x < src.cols-bsize; x+=bstep) {
        cv::Rect roi = cv::Rect(x, 0, bsize, src.rows);
        cv::Mat block = src(roi);
        cv::Mat hist;
        int hist_size = 8;
        cv::calcHist(&block, 1, 0, cv::Mat(), hist, 1, &hist_size, 0);
        cv::Mat prob = hist / block.total();
        low_intensities_prob.push_back(prob.at<float>(0, 0));
    }

    cv::Mat acc_sum = cv::Mat::zeros(cv::Size(low_intensities_prob.size(), 1), CV_32F);
    acc_sum.at<float>(0, 0) = low_intensities_prob[0];

    for (int i = 1; i < low_intensities_prob.size() ; i++) {
        acc_sum.at<float>(0, i) = low_intensities_prob[i] + acc_sum.at<float>(0, i-1);
    }

    cv::normalize(acc_sum, acc_sum, 0, 1, cv::NORM_MINMAX);

    std::vector<float> v = image_util::mat2vector<float>(acc_sum);
    std::vector<float>::iterator low = std::lower_bound(v.begin(), v.end(), 0.7);
    uint32_t rx = (low - v.begin()) * bstep + cols_mid;

    cv::Rect new_roi = cv::Rect(0, 0, rx, src.rows);
    dst_arr.create(cv::Size(new_roi.width, new_roi.height), src.type());
    cv::Mat dst = dst_arr.getMat();
    src(new_roi).copyTo(dst);
}

cv::Mat preprocessing::extract_cartesian_mask2(const cv::Mat& sonar_image, const cv::Mat& mask, float alpha) {

    // calculate the proportional mean of each image row
    std::vector<float> row_mean(sonar_image.rows, 0);
    for (size_t i = 0; i < sonar_image.rows; i++) {
        double value = cv::sum(sonar_image.row(i))[0] / cv::countNonZero(mask.row(i));
        row_mean[sonar_image.rows - 1 - i] = std::isnan(value) ? 0 : value;
    }

    // accumulative sum
    std::vector<float> accum_sum_up(row_mean.size()), accum_sum_down(row_mean.size());
    std::partial_sum(row_mean.begin(), row_mean.end(), accum_sum_up.begin());
    std::partial_sum(row_mean.rbegin(), row_mean.rend(), accum_sum_down.begin());

    // threshold
    float min_up   = *std::min_element(accum_sum_up.begin(), accum_sum_up.end());
    float max_up   = *std::max_element(accum_sum_up.begin(), accum_sum_up.end());
    float min_down = *std::min_element(accum_sum_down.begin(), accum_sum_down.end());
    float max_down = *std::max_element(accum_sum_down.begin(), accum_sum_down.end());

    float thresh_up   = alpha * (max_up - min_up) + min_up;
    float thresh_down = alpha * (max_down - min_down) + min_down;
    std::replace_if(accum_sum_up.begin(), accum_sum_up.end(), std::bind2nd(std::less<float>(), thresh_up), 0.0);
    std::replace_if(accum_sum_down.begin(), accum_sum_down.end(), std::bind2nd(std::less<float>(), thresh_down), 0.0);

    // roi the mask (y axis)
    std::vector<float>::iterator pos_up   = std::find_if (accum_sum_up.begin(), accum_sum_up.end(), std::bind2nd(std::greater<float>(), 0));
    std::vector<float>::iterator pos_down = std::find_if (accum_sum_down.begin(), accum_sum_down.end(), std::bind2nd(std::greater<float>(), 0));

    uint32_t y_bottom = std::distance(accum_sum_up.begin(), pos_up) + 1;
    uint32_t y_top    = std::distance(accum_sum_down.begin(), pos_down) + 1;

    mask.rowRange(0, y_top).setTo(cv::Scalar(0));
    mask.rowRange(mask.rows - y_bottom, mask.rows).setTo(cv::Scalar(0));

    return mask;
}

cv::Mat preprocessing::extract_cartesian_mask (   const cv::Mat& sonar_image, const cv::Mat& mask,
                                                  float alpha) {

    // calculate the proportional mean of each image row
    std::vector<float> row_mean(sonar_image.rows, 0);
    for (size_t i = 0; i < sonar_image.rows; i++) {
        double value = cv::sum(sonar_image.row(i))[0] / cv::countNonZero(mask.row(i));
        row_mean[sonar_image.rows - 1 - i] = std::isnan(value) ? 0 : value;
    }

    // accumulative sum
    std::vector<float> accum_sum_up(row_mean.size()), accum_sum_down(row_mean.size());
    std::partial_sum(row_mean.begin(), row_mean.end(), accum_sum_up.begin());
    std::partial_sum(row_mean.rbegin(), row_mean.rend(), accum_sum_down.begin());

    // threshold
    float min_up   = *std::min_element(accum_sum_up.begin(), accum_sum_up.end());
    float max_up   = *std::max_element(accum_sum_up.begin(), accum_sum_up.end());
    float min_down = *std::min_element(accum_sum_down.begin(), accum_sum_down.end());
    float max_down = *std::max_element(accum_sum_down.begin(), accum_sum_down.end());

    float thresh_up   = alpha * (max_up - min_up) + min_up;
    float thresh_down = alpha * (max_down - min_down) + min_down;
    std::replace_if(accum_sum_up.begin(), accum_sum_up.end(), std::bind2nd(std::less<float>(), thresh_up), 0);
    std::replace_if(accum_sum_down.begin(), accum_sum_down.end(), std::bind2nd(std::less<float>(), thresh_down), 0);

    // roi the mask (y axis)
    std::vector<float>::iterator pos_up   = std::find_if (accum_sum_up.begin(), accum_sum_up.end(), std::bind2nd(std::greater<float>(), 0));
    std::vector<float>::iterator pos_down = std::find_if (accum_sum_down.begin(), accum_sum_down.end(), std::bind2nd(std::greater<float>(), 0));

    uint32_t y_bottom = std::distance(accum_sum_up.begin(), pos_up) + 1;
    uint32_t y_top    = std::distance(accum_sum_down.begin(), pos_down) + 1;

    cv::Mat dst = mask.clone();
    dst.rowRange(0, y_top).setTo(0);
    dst.rowRange(dst.rows - y_bottom, dst.rows).setTo(0);

    // roi the mask (x axis)
    cv::Mat mask_sum;
    cv::reduce(dst / 255, mask_sum, 0, CV_REDUCE_SUM, CV_32S);
    int max = *std::max_element(mask_sum.begin<int>(), mask_sum.end<int>());
    for (size_t i = 0; i < mask_sum.cols; i++) {
        if(mask_sum.at<int>(i) != max)
            dst.col(i).setTo(0);
    }

    return dst;
}

std::vector<cv::Point> preprocessing::find_biggest_contour(cv::InputArray src_arr) {
    cv::Mat src = src_arr.getMat();
    cv::Mat im_contours;

    if (src_arr.depth() == CV_32F) {
        src.convertTo(im_contours, CV_8U, 255.0);
    }
    else {
        src.copyTo(im_contours);
    }

    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(im_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    int last_size_area = 0;
    int biggest_index = -1;
    for( int i = 0; i < contours.size(); i++ ) {
        cv::RotatedRect box = cv::minAreaRect(cv::Mat(contours[i]));
        float size_area = box.size.area();
        if (size_area > last_size_area) {
            last_size_area = size_area;
            biggest_index = i;
        }
    }

    std::vector<cv::Point> contour;
    cv::convexHull(cv::Mat(contours[biggest_index]), contour, false );
    return contour;
}

} /* namespace sonar_processing */
