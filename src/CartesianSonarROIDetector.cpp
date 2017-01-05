#include <cstdio>
#include "base/Plot.hpp"
#include "CartesianSonarROIDetector.hpp"

namespace sonar_processing {

int CartesianSonarROIDetector::GetStartBin(float cutoff) {
    int beam = sonar_holder_.beam_count() / 2;
    cv::Point2f first_point = sonar_holder_.cart_center_point(initial_bin_, beam);
    cv::Point2f last_point = sonar_holder_.cart_center_point(sonar_holder_.bin_count()/2, beam);

    size_t total_lines = floor(first_point.y-last_point.y);
    size_t first_y = (size_t)first_point.y;
    size_t last_y = (size_t)last_point.y;

    cv::Mat cart_image = sonar_holder_.cart_image();

    std::vector<float> mean_values(total_lines, 0);

    for (size_t line = first_y, i = 0; line > last_y && i < total_lines; line--, i++) {
        int x0, x1;
        sonar_holder_.cart_line_limits(line, x0, x1);
        if (x0 >= 0 && x1 >= 0 && (x1-x0) > 0) {
            mean_values[i] = cv::mean(cart_image.row(line).colRange(x0, x1))[0];
        }
    }

    int cutoff_index = GetCutOffIndex(mean_values, cutoff);
    int line = first_y - cutoff_index;
    int polar_index = sonar_holder_.cart_to_polar_index(sonar_holder_.cart_size().width / 2, line);
    return sonar_holder_.index_to_bin(polar_index);
}

int CartesianSonarROIDetector::GetFinalBin(float cutoff) {
    int beam = sonar_holder_.beam_count() / 2;

    cv::Point2f last_point = sonar_holder_.cart_center_point(last_final_bin_, beam);
    cv::Point2f first_point = sonar_holder_.cart_center_point(first_final_bin_,  beam);

    int first_line = last_point.y;
    int last_line  = first_point.y;
    int total_lines = last_line - first_line;

    cv::Mat cart_image = sonar_holder_.cart_image();
    std::vector<float> low_probs(total_lines, 0.0);

    for (size_t y = first_line, i = 0; y <= last_line && i < total_lines; y++, i++) {
        std::vector<float> values;

        int x0, x1;
        sonar_holder_.cart_line_limits(y, x0, x1);

        if (x0 >= 0 && x1 >= 0 && (x1-x0) > 0) {
            int count = 0;
            for (int x = x0; x <= x1; x++) {
                if (cart_image.at<float>(y, x) < low_probs_thresh_) count++;
            }
            low_probs[i] = count / (float)(x1 - x0);
        }
    }

    cv::Mat1f low_probs_mat(low_probs);

    cv::blur(low_probs_mat, low_probs_mat, cv::Size(25, 25));
    cv::normalize(low_probs_mat, low_probs_mat, cv::NORM_MINMAX);

    int cutoff_index = GetCutOffIndex(low_probs, cutoff);
    int line = first_line + cutoff_index;
    int polar_index = sonar_holder_.cart_to_polar_index(sonar_holder_.cart_size().width / 2, line);

    return sonar_holder_.index_to_bin(polar_index);
}

} /* namespace sonar_processing */
