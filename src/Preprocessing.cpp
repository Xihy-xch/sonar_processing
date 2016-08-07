#include "base/Plot.hpp"
#include "sonar_target_tracking/Preprocessing.hpp"
#include "sonar_target_tracking/ImageUtils.hpp"
#include "sonar_target_tracking/Utilities.hpp"
#include "sonar_target_tracking/third_party/spline.h"

namespace sonar_target_tracking {

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
    std::vector<cv::Vec4i> hierarchy;

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

cv::Rect preprocessing::calc_horiz_roi(cv::Mat src) {
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

    float thresh = ((min + max) / 2.0) * 0.25;
    cv::Mat bin;
    cv::threshold(acc_sum, bin, thresh, 1.0, cv::THRESH_BINARY);

    std::vector<float> bin_vec = image_utils::mat2vector<float>(bin);
    uint32_t new_x = std::find(bin_vec.begin(), bin_vec.end(), 1) - bin_vec.begin();

    return cv::Rect(cv::Point(new_x, 0), cv::Size(acc_sum.cols - new_x, src.rows));
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
            dsum += abs(mat.at<float>(y, x) - mat.at<float>(y,x+1));
        }
    }
    return dsum / src.total();
}

std::vector<std::vector<cv::Point> > preprocessing::find_contours(cv::Mat src) {
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat im_contours;
    src.copyTo(im_contours);

    cv::findContours(im_contours, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    return contours;

}

std::vector<cv::Point> preprocessing::find_biggest_contour(cv::Mat src) {
    std::vector<std::vector<cv::Point> > contours = find_contours(src);

    std::vector<cv::Point> biggest_contour;
    int last_size = 0;
    for( int i = 0; i < contours.size(); i++ ) {

        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        int size = rc.width * rc.height;
        if (size > last_size) {
            last_size = size;
            biggest_contour = contours[i];
        }
    }

    return biggest_contour;
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

std::vector<std::vector<cv::Point> > preprocessing::find_contours_and_filter(cv::Mat src, cv::Size min_size) {
    std::vector<std::vector<cv::Point> > filtering_contours;
    std::vector<std::vector<cv::Point> > contours = find_contours(src);

    int min_area = min_size.width * min_size.height;

    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        if (cv::contourArea(contours[i]) > min_area &&
            rc.width > min_size.width &&
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

cv::Mat preprocessing::remove_ground_distance(cv::Mat src, cv::Rect& horiz_roi) {
    cv::Mat eqhist = cv::Mat::zeros(src.size(), src.type());
    image_utils::equalize_histogram_32f(src, eqhist);
    cv::medianBlur(eqhist, eqhist, 5);
    cv::Mat result;
    src(horiz_roi = preprocessing::calc_horiz_roi(eqhist)).copyTo(result);
    return result;
}

cv::Mat preprocessing::remove_ground_distance_accurate(cv::Mat src, cv::Rect& horiz_roi) {
    cv::Mat mat = remove_ground_distance(src, horiz_roi);

    std::vector<uint32_t> ground_distance_line = preprocessing::compute_ground_distance_line(mat);
    horiz_roi.x = horiz_roi.x + *std::max_element(ground_distance_line.begin(), ground_distance_line.end());
    horiz_roi.width = src.cols - horiz_roi.x;
    src(horiz_roi).copyTo(mat);
    return mat;
}

std::vector<uint32_t> preprocessing::compute_ground_distance_line(cv::Mat mat) {
    std::vector<double> X;
    std::vector<double> Y;

    int row_step = floor((mat.rows-1) / 2);

    for (int row = 0; row < mat.rows; row += row_step) {
        X.push_back(find_first_higher(mat, row));
        Y.push_back(row);
    }

    std::vector<uint32_t> cols;
    uint32_t y = 0;

    tk::spline spline;
    spline.set_points(Y, X);

    while (y < mat.rows) {
        cols.push_back((uint32_t)utilities::clip(spline((double)y++), 0, mat.rows-1));
    }

    return cols;
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

std::vector<double> preprocessing::background_features_estimation(cv::Mat src, uint32_t bsize) {
    cv::Rect roi;
    cv::Mat mat = remove_ground_distance_accurate(src, roi);
    image_utils::equalize_histogram_32f(mat, mat);
    image_utils::clahe_32f(mat, mat);

    double mean_sum = 0;
    double stddev_sum = 0;
    double block_count = 0;

    for (int y = 0; y < mat.rows-bsize; y+=bsize) {
        for (int x = 0; x < mat.cols-bsize; x+=bsize) {
            roi = cv::Rect(x, y, bsize, bsize);
            cv::Mat block;
            mat(roi).copyTo(block);
            cv::Scalar mean, stddev;
            cv::meanStdDev(block, mean, stddev);
            mean_sum += mean[0];
            stddev_sum += stddev[0];
            block_count += 1;
        }
    }

    std::vector<double> features;
    features.push_back(mean_sum / block_count);
    features.push_back(stddev_sum / block_count);
    return features;

}

void preprocessing::background_features_difference(cv::InputArray src_arr, cv::OutputArray dst_arr, std::vector<double> features, uint32_t bsize) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();
    dst.setTo(0);

    for (int y = 0; y < src.rows-bsize; y++) {
        for (int x = 0; x < src.cols-bsize; x++) {
            cv::Rect roi = cv::Rect(x, y, bsize, bsize);
            cv::Mat block;
            src(roi).copyTo(block);
            cv::Scalar mean, stddev;
            cv::meanStdDev(block, mean, stddev);
            double mean_diff = mean[0] - features[0];
            double stddev_diff = 0;
            dst.at<float>(y, x) = sqrt(mean_diff * mean_diff + stddev_diff * stddev_diff);
        }
    }
}

std::vector<std::vector<cv::Point> > preprocessing::target_detect_by_high_intensities(cv::InputArray src_arr) {
    cv::Mat src = src_arr.getMat();

    cv::Mat mat = cv::Mat::zeros(src.size(), src.type());
    image_utils::equalize_histogram_32f(src, mat);
    image_utils::clahe_32f(mat, mat);

    cv::imshow("mat", mat);

    cv::Mat bin;
    cv::normalize(mat, bin, 0, 1, cv::NORM_MINMAX);
    cv::boxFilter(bin, bin, CV_32F, cv::Size(5, 5));
    cv::threshold(bin, bin, 0.7, 1.0, cv::THRESH_BINARY);

    mat.setTo(0);
    preprocessing::mean_horiz_difference_thresholding(bin, mat, 10, 0.3, 0.2);

    mat.convertTo(mat, CV_8UC1, 255);
    std::vector<std::vector<cv::Point> > contours;

    contours = adaptative_find_contours_and_filter(mat, 0.05, 0.2, 0.2);

    mat.setTo(0);
    cv::drawContours(mat, contours, -1, cv::Scalar(255), CV_FILLED);

    cv::morphologyEx(mat, mat, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    cv::morphologyEx(mat, mat, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);

    return adaptative_find_contours_and_filter(mat, 1, 1, 1);
}

void preprocessing::contrast_filter(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();
    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    src.copyTo(dst);

    const double max_entropy_thesh = 7.5;
    const uint32_t bsize = src.cols/2;

    for (int x = 0, i = 0; x < src.cols; x+=bsize, i++) {
        uint32_t r = (src.cols > x + bsize) ? bsize : src.cols - x;
        cv::Rect roi = cv::Rect(x, 0, r, src.rows);
        cv::Mat block = dst(roi);
        image_utils::adaptative_clahe(block, block, cv::Size(8, 8), max_entropy_thesh);
        cv::blur(block, block, cv::Size(3, 3));
    }
}

void preprocessing::gradient_filter(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    cv::Mat Gx, Gy;
    cv::Mat Gx2, Gy2;

    cv::Sobel( src, Gx, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gx, Gx2 );

    cv::Sobel( src, Gy, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gy, Gy2 );

    cv::addWeighted( Gx2, 0.5, Gy2, 0.5, 0, dst );
}

void preprocessing::weak_target_thresholding(cv::InputArray src_arr, cv::InputArray src_hc_arr, cv::OutputArray dst_arr) {
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

    for (int y = 0; y < src_height; y+=max_win_height/2) {
        for (int x = 0; x < src_width; x+=max_win_width/2) {

            uint32_t win_width = (src_width > x + max_win_width) ? max_win_width : src_width - x;
            uint32_t win_height = (src_height > y + max_win_height) ? max_win_height : src_height - y;

            if (win_width < max_win_width / 2) continue;
            if (win_height < max_win_height / 2) continue;

            cv::Rect win_rect = cv::Rect(x, y, win_width, win_height);

            cv::Mat max_sum;
            src_hc(win_rect).copyTo(max_sum);
            cv::threshold(max_sum, max_sum, high_intensities_thesh, 1, cv::THRESH_BINARY);

            if ((cv::sum(max_sum)[0] / max_sum.total()) > high_intensities_mean_thresh) {
                cv::Mat win;
                src(win_rect).convertTo(win, CV_8U, 255);
                cv::blur(win, win, cv::Size(15, 15));

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

std::vector<std::vector<cv::Point> > preprocessing::find_target_contours(cv::InputArray src_arr) {
    cv::Mat src, src_8u, src_hc;
    src = src_arr.getMat();
    src.convertTo(src_8u, CV_8U, 255);
    contrast_filter(src_8u, src_hc);

    cv::Mat hig_mask;
    weak_target_thresholding(src, src_hc, hig_mask);

    std::vector<std::vector<cv::Point> > contours = find_contours_and_filter(hig_mask, cv::Size(src.cols * 0.025, src.rows * 0.025));

    hig_mask.setTo(0);
    cv::drawContours(hig_mask, contours, -1, cv::Scalar(255), CV_FILLED);

    cv::morphologyEx(hig_mask, hig_mask, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)),
                     cv::Point(-1, -1), 1);

    contours = find_contours(hig_mask);

    if (contours.empty()) {
        return std::vector<std::vector<cv::Point> >();
    }

    std::vector<cv::Rect> rects;
    cv::Mat mean_vals = cv::Mat::zeros(cv::Size(1, contours.size()), CV_32F);
    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        rects.push_back(rc);
        if (rc.width > 50 && rc.height > 50) mean_vals.at<float>(i, 1) = cv::mean(src_8u(rc))[0];
    }

    double min, max;
    cv::minMaxLoc(mean_vals, &min, &max);

    double thresh = (max + min) / 2;

    hig_mask.setTo(0);
    for( int i = 0; i < mean_vals.rows; i++ ) {
        if (cv::mean(src_8u(rects[i]))[0] > thresh) {
            cv::rectangle(hig_mask, rects[i], cv::Scalar(255), CV_FILLED);
        }
    }

    cv::morphologyEx(hig_mask, hig_mask, cv::MORPH_CLOSE,
                 cv::getStructuringElement(cv::MORPH_RECT,cv::Size(40, 40)),
                 cv::Point(-1, -1), 1);

    cv::Mat hig_image = cv::Mat::zeros(src_8u.size(), src_8u.type());

    contours = find_contours(hig_mask);
    std::vector<std::vector<cv::Point> > contours_result;
    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        if (rc.width > 30 && rc.height > 30) contours_result.push_back(contours[i]);
    }

    return contours_result;
}

} /* namespace sonar_target_tracking */
