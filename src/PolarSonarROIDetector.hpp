#ifndef sonar_processing_PolarSonarROIDetector_hpp
#define sonar_processing_PolarSonarROIDetector_hpp

#include "sonar_processing/ROI.hpp"
#include "sonar_processing/SonarHolder.hpp"
#include "sonar_processing/BasicOperations.hpp"

namespace sonar_processing {

class PolarSonarROIDetector : public SonarROIDetector {
public:

    PolarSonarROIDetector(const SonarHolder& sonar_holder, int initial_bin = 1) 
        : SonarROIDetector(sonar_holder, initial_bin)
    {
    }

    virtual ~PolarSonarROIDetector() {
    }

protected:

    int GetStartBin(float cutoff);
    int GetFinalBin(float cutoff);

};

} /* namespace sonar_processing */


#endif /* sonar_processing_PolarSonarROIDetector_hpp */
