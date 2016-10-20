#include "base/Plot.hpp"
#include "sonar_target_tracking/Utils.hpp"
#include "ROI.hpp"

namespace sonar_target_tracking {

namespace roi {

SonarROIDetector::SonarROIDetector(
    const SonarHolder& sonar_holder,
    int initial_bin)
    : sonar_holder_(sonar_holder)
    , initial_bin_(initial_bin)
{
}

SonarROIDetector::~SonarROIDetector() {

}

void SonarROIDetector::GetBinsOfInterest(int& start_bin, int& final_bin, float start_bin_cutoff, float final_bin_cutoff) {
    start_bin = GetStartBin(start_bin_cutoff);
    final_bin = GetFinalBin(final_bin_cutoff);
}

int SonarROIDetector::GetStartBin(float cutoff) {
    std::vector<float> averages;

    basic_operations::average_lines(sonar_holder_, initial_bin_, sonar_holder_.bin_count() / 2, averages);
    return GetCutOffBin(averages, initial_bin_, cutoff);
}

int SonarROIDetector::GetFinalBin(float cutoff) {

    int first_bin = sonar_holder_.bin_count() - (sonar_holder_.bin_count() / 3);
    int last_bin = sonar_holder_.bin_count() - 2;
    int beam = sonar_holder_.beam_count() / 2;

    std::vector<float> low_probs(last_bin - first_bin, 0);
    std::vector<float> high_probs(last_bin - first_bin, 0);

    for (int bin = first_bin, i = 0; bin < last_bin; bin++, i++) {
        std::vector<float> values;
        basic_operations::line_values(sonar_holder_, sonar_holder_.index_at(beam, bin), values);
        float low_prob, high_prob;
        EvalProbs(values, 0.05, low_probs[i], high_probs[i]);
    }

    cv::Mat1f low_probs_mat(low_probs);
    cv::blur(low_probs_mat, low_probs_mat, cv::Size(25, 25));
    cv::normalize(low_probs_mat, low_probs_mat, 0, 1, cv::NORM_MINMAX);

    return GetCutOffBin(low_probs, first_bin, cutoff);
}

int SonarROIDetector::GetCutOffBin(std::vector<float> values, int offset, float cutoff) {
    std::vector<float> accsum;
    utils::accumulative_sum<float>(values, accsum);

    std::vector<float> accsum_norm(accsum.size());

    cv::Mat1f accsum_mat(accsum);
    cv::Mat1f accsum_norm_mat(accsum_norm);

    cv::normalize(accsum_mat, accsum_norm_mat, 0, 1, cv::NORM_MINMAX);

    return std::upper_bound(accsum_norm.begin(), accsum_norm.end(), cutoff) - accsum_norm.begin() + offset;
}

void SonarROIDetector::EvalProbs(std::vector<float> values, float thresh, float& low_prob, float& high_prob) {
    int low_count = 0;
    int high_count = 0;

    for (int i = 0; i < values.size(); i++) {
        if (values[i] < thresh) low_count++; else high_count++;
    }

    low_prob = low_count / (float)values.size();
    high_prob = high_count / (float)values.size();
}

} /* namespace roi */

} /* namespace sonar_target_tracking  */
