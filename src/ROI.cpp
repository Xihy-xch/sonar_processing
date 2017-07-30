#include "ROI.hpp"
#include "PolarSonarROIDetector.hpp"
#include "CartesianSonarROIDetector.hpp"

namespace sonar_processing {

/* SonarROIDetector Implementation */
int SonarROIDetector::GetCutOffIndex(std::vector<float> values, float cutoff) {
    std::vector<float> accsum;
    utils::accumulative_sum<float>(values, accsum);

    std::vector<float> accsum_norm(accsum.size());
    cv::Mat1f accsum_mat(accsum);
    cv::Mat1f accsum_norm_mat(accsum_norm);

    cv::normalize(accsum_mat, accsum_norm_mat, 0, 1, cv::NORM_MINMAX);

    return std::upper_bound(accsum_norm.begin(), accsum_norm.end(), cutoff) - accsum_norm.begin();
}

void SonarROIDetector::GetLowProbs(std::vector<float> values, float thresh, float& prob) {
    int count = 0;

    for (int i = 0; i < values.size(); i++) {
        if (values[i] < thresh) count++;
    }

    prob = count / (float)values.size();
}

/* End SonarROIDetector Implementation */
    
namespace roi {
    
void polar::bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin) {
    sonar_processing::PolarSonarROIDetector roi_detector(sonar_holder, initial_bin);
    roi_detector.GetBinsOfInterest(start_bin, final_bin, 0.2, 0.5);
}

void cartesian::bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin) {
    sonar_processing::CartesianSonarROIDetector roi_detector(sonar_holder, initial_bin);
    roi_detector.GetBinsOfInterest(start_bin, final_bin, 0.12, 0.5);
}

int cartesian::find_start_bin(const SonarHolder& sonar_holder, float cutoff, int initial_bin) {
    sonar_processing::CartesianSonarROIDetector roi_detector(sonar_holder, initial_bin);
    return roi_detector.GetStartBin(cutoff);
}

} /* namespace sonar_processing */

} /* namespace roi */
