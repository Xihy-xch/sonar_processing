#include "Utils.hpp"
#include "PolarSonarROIDetector.hpp"

namespace sonar_processing {

int PolarSonarROIDetector::GetStartBin(float cutoff) {
    std::vector<float> averages;
    basic_operations::average_lines(sonar_holder_, initial_bin_, sonar_holder_.bin_count() / 2, averages);
    return GetCutOffIndex(averages, cutoff) + initial_bin_;
}

int PolarSonarROIDetector::GetFinalBin(float cutoff) {

    int first_bin = sonar_holder_.bin_count() - (sonar_holder_.bin_count() / 3);
    int last_bin = sonar_holder_.bin_count() - 2;
    int beam = sonar_holder_.beam_count() / 2;

    std::vector<float> low_probs(last_bin - first_bin, 0);

    for (int bin = first_bin, i = 0; bin < last_bin; bin++, i++) {
        std::vector<float> values;
        basic_operations::line_values(sonar_holder_, sonar_holder_.index_at(beam, bin), values);
        GetLowProbs(values, 0.05, low_probs[i]);
    }

    cv::Mat1f low_probs_mat(low_probs);
    cv::Mat1f blurred;
    low_probs_mat.copyTo(low_probs_mat, blurred);
    cv::blur(blurred, blurred, cv::Size(25, 25));
    cv::normalize(low_probs_mat, low_probs_mat, 0, 1, cv::NORM_MINMAX);

    return GetCutOffIndex(low_probs, cutoff) + first_bin;
}

} /* namespace sonar_processing */
