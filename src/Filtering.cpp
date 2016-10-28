#include "sonar_processing/Filtering.hpp"

namespace sonar_processing {

namespace filtering {

void FilterApplier::PerformConvolution(std::vector<float>& dst, const std::vector<float>& kernel, int ksize) const {
    dst.assign(sonar_holder_.total_elements(), 0);
    for (int index= 0; index < sonar_holder_.total_elements(); index++) {
        if (validate_index(index)) {
            std::vector<int> neighbor_indices;
            basic_operations::neighborhood(sonar_holder_, index, ksize, neighbor_indices);
            if (validate_neighborhood_indices(neighbor_indices)) {
                std::vector<float> values;
                sonar_holder_.values(neighbor_indices, values);
                cv::Mat kernel_mat = cv::Mat(kernel).reshape(1, ksize);
                cv::Mat values_mat = cv::Mat(values).reshape(1, ksize);
                dst[index] = ApplyKernel(values_mat, kernel_mat);
            }
        }
    }
}

float FilterApplier::ApplyKernel(cv::Mat src, cv::Mat kernel) const {

    float result = 0;
    for (int y = 0; y < kernel.rows; y++) {
        for (int x = 0; x < kernel.cols; x++) {
            result += kernel.at<float>(y, x) * src.at<float>(y, x);
        }
    }

    return result;
}

} /* namespace filtering */

} /* namespace sonar_processing */
