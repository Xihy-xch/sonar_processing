#ifndef CartesianSonarROIDetector_hpp
#define CartesianSonarROIDetector_hpp

#include "sonar_processing/ROI.hpp"
#include "sonar_processing/SonarHolder.hpp"

namespace sonar_processing {

class CartesianSonarROIDetector : public SonarROIDetector {
public:

    CartesianSonarROIDetector(
        const SonarHolder& sonar_holder,
        int initial_bin = 1,
        int first_final_bin_ratio = 3,
        int last_final_bin_offset = 3,
        float low_probs_ratio = 16.0)
        : SonarROIDetector(sonar_holder, initial_bin)
    {
        last_final_bin_ = sonar_holder_.bin_count() - last_final_bin_offset;
        first_final_bin_ = sonar_holder_.bin_count() - (sonar_holder_.bin_count() / first_final_bin_ratio);
        low_probs_thresh_ = 1.0 / low_probs_ratio;
    }

    virtual ~CartesianSonarROIDetector() {
    }

protected:

    int GetStartBin(float cutoff);
    int GetFinalBin(float cutoff);

    int last_final_bin_;
    int first_final_bin_;
    float low_probs_thresh_;

};

} /* namespace sonar_processing */


#endif /* CartesianSonarROIDetector_hpp */
