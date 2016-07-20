#include "sonar_target_tracking/Preprocessing.hpp"

namespace sonar_target_tracking {
    
cv::Rect preprocessing::calc_horiz_roi(cv::Mat src) {
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

} /* namespace sonar_target_tracking */
