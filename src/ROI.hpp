#ifndef sonar_target_tracking_SonarROI_hpp
#define sonar_target_tracking_SonarROI_hpp

#include <stdio.h>
#include "sonar_target_tracking/SonarHolder.hpp"
#include "sonar_target_tracking/BasicOperations.hpp"

namespace sonar_target_tracking {

namespace roi {

class SonarROIDetector {
public:

    SonarROIDetector(const SonarHolder& sonar_holder, int initial_bin = 1);
    virtual ~SonarROIDetector();

    void GetBinsOfInterest(int& start_bin, int& final_bin, float start_bin_cuttoff = 0.15, float final_bin_cutoff = 0.5);

private:

    int GetStartBin(float cutoff);
    int GetFinalBin(float cutoff);
    int GetCutOffBin(std::vector<float> values, int offset, float cutoff);

    void EvalProbs(std::vector<float> values, float thresh, float& low_prob, float& high_prob);


    const SonarHolder& sonar_holder_;
    int initial_bin_;
};

inline void bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin = 1) {
    SonarROIDetector roi_detector(sonar_holder, initial_bin);
    roi_detector.GetBinsOfInterest(start_bin, final_bin);
}

} /* namespace roi */

} /* namespace sonar_target_tracking  */


#endif /* sonar_target_tracking_SonarROI_hpp */
