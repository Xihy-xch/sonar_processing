#ifndef sonar_processing_SonarROI_hpp
#define sonar_processing_SonarROI_hpp

#include <vector>
#include <opencv2/opencv.hpp>
#include "sonar_processing/Utils.hpp"
#include "sonar_processing/SonarHolder.hpp"

namespace sonar_processing {

class SonarROIDetector {
public:
    SonarROIDetector(
        const SonarHolder& sonar_holder,
        int initial_bin)
        : sonar_holder_(sonar_holder)
        , initial_bin_(initial_bin)
    {
    }

    virtual ~SonarROIDetector()
    {
    }

    void GetBinsOfInterest(int& start_bin, int& final_bin, float start_bin_cutoff, float final_bin_cutoff) {
        start_bin = GetStartBin(start_bin_cutoff);
        final_bin = GetFinalBin(final_bin_cutoff);
    }

    virtual int GetStartBin(float cutoff) = 0;
    virtual int GetFinalBin(float cutoff) = 0;

protected:
    virtual int GetCutOffIndex(std::vector<float> values, float cutoff);
    virtual void GetLowProbs(std::vector<float> values, float thresh, float& prob);

    const SonarHolder& sonar_holder_;
    int initial_bin_;
};

namespace roi {

namespace polar {

void bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin = 1);

} /* namespace polar */

namespace cartesian {

void bins_of_interest(const SonarHolder& sonar_holder, int& start_bin, int& final_bin, int initial_bin = 1);

int find_start_bin(const SonarHolder& sonar_holder, float cutoff=0.03, int initial_bin=1);

} /* namespace cartesian */

} /* namespace roi */

} /* namespace sonar_processing  */


#endif /* sonar_processing_SonarROI_hpp */
