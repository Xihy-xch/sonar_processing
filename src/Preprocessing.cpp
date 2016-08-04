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
    std::vector<float>::iterator it = std::find(bin_vec.begin(), bin_vec.end(), 1);

    cv::Rect rc(cv::Point(0, 0), src.size());
    rc.x = it - bin_vec.begin();
    rc.width = acc_sum.cols - rc.x;

    return rc;
}

double preprocessing::horiz_deriv(cv::Mat src) {
    CV_Assert(src.depth() == CV_32F);
    float dsum = 0;
    for (int y = 0; y < src.rows-1; y++) {
        for (int x = 0; x < src.cols-1; x++) {
            float d = src.at<float>(y, x) - src.at<float>(y,x+1);
            dsum += abs(d);
        }
    }
    return dsum;
}

std::vector<cv::Point> preprocessing::find_biggest_contour(cv::Mat src) {
    std::vector<std::vector<cv::Point> > contours;

    cv::Mat im_contours;
    src.copyTo(im_contours);
    cv::findContours(im_contours, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

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

std::vector<std::vector<cv::Point> > preprocessing::find_contours_and_filter(cv::Mat src, double area_factor, double width_factor, double height_factor) {
    std::vector<std::vector<cv::Point> > contours, filtering_contours, nonzero_area_contours;

    cv::Mat im_contours;
    src.copyTo(im_contours);

    cv::findContours(im_contours, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

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

void preprocessing::adaptative_threshold(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    cv::Mat src = src_arr.getMat();

    dst_arr.create(src.size(), src.type());
    cv::Mat dst = dst_arr.getMat();

    cv::imshow("source", src);

    cv::Mat mask, mat;
    src.convertTo(mat, CV_8U, 255);

    cv::blur(mat, mat, cv::Size(7, 7));

    double thresh;
    thresh = image_utils::otsu_thresh_8u(mat) * 0.9;
    cv::threshold(mat, mask, thresh, 255, cv::THRESH_BINARY);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11, 11)), cv::Point(-1, -1), 2);
    cv::morphologyEx(mask, mask, cv::MORPH_ERODE, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7, 7)), cv::Point(-1, -1), 2);

    std::vector<cv::Point> contour = preprocessing::find_biggest_contour(mask);
    mask.setTo(0);
    cv::Rect rc = cv::boundingRect(cv::Mat(contour));

    image_utils::cv32f_clahe(src, src);
    src(rc).copyTo(mat);

    mat.convertTo(mat, CV_8U, 255);
    cv::blur(mat, mat, cv::Size(7, 7));
    thresh = image_utils::otsu_thresh_8u(mat) * 1.5;
    cv::threshold(mat, mask, thresh, 255, cv::THRESH_BINARY);

    src(rc).copyTo(mat, mask);
    mat.convertTo(mat, CV_8U, 255);
    cv::morphologyEx(mat, mat, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5, 5)), cv::Point(-1, -1), 2);
    thresh = image_utils::otsu_thresh_8u(mat) * 2;
    cv::threshold(mat, mask, thresh, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point> > contours = preprocessing::find_contours_and_filter(mask, 0.1, 0.2, 0.2);

    mask.setTo(0);
    cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5, 5)), cv::Point(-1, -1), 2);

    src(rc).copyTo(mat, mask);
    cv::imshow("result", mat);
    cv::imshow("mask", mask);

    // contours = preprocessing::find_contours_and_filter(mask, 0.75, 0.75, 0.75);
    // mask.setTo(0);
    // for( int i = 0; i < contours.size(); i++ ) {
    //     cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
    //     mask(rc) = 255;
    // }
    //
    //
    // cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);

    // cv::imshow("mask", mask);
    // src.copyTo(mat, mask);
    // cv::imshow("mat", mat);

    // cv::drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
    // cv::imshow("mask 1", mask);

    // mat.setTo(0);
    // src.copyTo(mat, mask);
    // mat.convertTo(mat, CV_8U, 255);
    // cv::imshow("mat 1", mat);


    // cv::boxFilter(mat, mat, CV_8U, cv::Size(9, 9));
    // cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    //
    // thresh =  image_utils::otsu_thresh_8u(mat) * 1.4;
    // cv::threshold(mat, mask, thresh, 255, cv::THRESH_BINARY);
    // cv::imshow("mask 2", mask);
    //
    //
    // mat.setTo(0);
    // src.copyTo(mat, mask);
    // mat.convertTo(mat, CV_8U, 255);
    //
    // cv::boxFilter(mat, mat, CV_8U, cv::Size(7, 7));
    // cv::normalize(mat, mat, 0, 255, cv::NORM_MINMAX);
    //
    // thresh =  image_utils::otsu_thresh_8u(mat);
    // cv::threshold(mat, mask, thresh, 255, cv::THRESH_BINARY);
    // cv::imshow("mask 3", mask);
    // std::vector<std::vector<cv::Point> > contours = preprocessing::find_contours_and_filter(mask, min_blob_size);
    // cv::imshow("mask 4", mask);

}

void preprocessing::mean_horiz_deriv_threshold(cv::InputArray src_arr, cv::OutputArray dst_arr, uint32_t bsize,
                                               double mean_thresh, double horiz_deriv_thresh) {

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
            double d = horiz_deriv(block) / (bsize * bsize);

            if (m > mean_thresh && d < horiz_deriv_thresh) {
                dst(cv::Rect(x, y, bsize/4, bsize/4)).setTo(1.0);
            }
        }
    }
}

cv::Mat preprocessing::remove_ground_distance(cv::Mat src, cv::Rect& horiz_roi) {
    cv::Mat eqhist = cv::Mat::zeros(src.size(), src.type());
    image_utils::cv32f_equalize_histogram(src, eqhist);
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
    image_utils::cv32f_equalize_histogram(mat, mat);
    image_utils::cv32f_clahe(mat, mat);

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

} /* namespace sonar_target_tracking */
