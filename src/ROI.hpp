#ifndef sonar_processing_SonarROI_hpp
#define sonar_processing_SonarROI_hpp

#include <stdio.h>
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_processing/BasicOperations.hpp"

namespace sonar_processing {

namespace roi {

class SonarROIDetector {
public:

    enum ScanningType {
        kPolarScanning = 1,
        kCartesianScanning = 2
    };

    SonarROIDetector(const SonarHolder& sonar_holder, int initial_bin = 1, ScanningType scanning_type = kPolarScanning);
    virtual ~SonarROIDetector();

    void GetBinsOfInterest(int& start_bin, int& final_bin, float start_bin_cuttoff = 0.15, float final_bin_cutoff = 0.5);

private:

    int GetStartBinPolar(float cutoff);
    int GetFinalBinPolar(float cutoff);
    int GetCutOffBinPolar(std::vector<float> values, int offset, float cutoff);

    void EvalProbs(std::vector<float> values, float thresh, float& low_prob, float& high_prob);

    const SonarHolder& sonar_holder_;
    int initial_bin_;
    ScanningType scanning_type_;
};
    
namespace polar {

inline void bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin = 1) {
    SonarROIDetector roi_detector(sonar_holder, initial_bin);
    roi_detector.GetBinsOfInterest(start_bin, final_bin);
}

} /* namespace polar */

namespace cartesian {

} /* namespace cartesian */

} /* namespace roi */

} /* namespace sonar_processing  */


#endif /* sonar_processing_SonarROI_hpp */
