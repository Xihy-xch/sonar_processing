#ifndef sonar_processing_ScannerBase_hpp
#define sonar_processing_ScannerBase_hpp

#include <vector>
#include <opencv2/opencv.hpp>

namespace sonar_processing {

class SonarHolder;

class ScannerBase {

public:

    ScannerBase() {
    }

    ScannerBase(SonarHolder const* sonar_holder)
        : sonar_holder_(sonar_holder)
    {
    }

    void set_sonar_holder(SonarHolder const* sonar_holder) {
        sonar_holder_ = sonar_holder;
    }

    virtual void GetCartesianLine(int from_index, std::vector<int>& indices) = 0;

    virtual void GetCartesianLine(int from_index, std::vector<int>& indices, int total_elements = -1) = 0;

    virtual void GetCartesianLine(int from_index, cv::Point2f from_point, std::vector<int>& indices) = 0;

    virtual void GetCartesianLine(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements) = 0;

    virtual void GetCartesianColumn(int from_index, std::vector<int>& indices) = 0;

    virtual void GetCartesianColumn(int from_index, std::vector<int>& indices, int total_elements = -1) = 0;

    virtual void GetCartesianColumn(int from_index, cv::Point2f from_point, std::vector<int>& indices) = 0;

    virtual void GetCartesianColumn(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements = -1) = 0;

    virtual void GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices) = 0;

    virtual void GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices, std::vector<cv::Point2f>& points) = 0;

    virtual void GetIntersectionLine(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) = 0;

    virtual void GetIntersectionColumn(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) = 0;

protected:
    SonarHolder const *sonar_holder_;
};

} /* namespace sonar_processing */

#endif /* sonar_processing_ScannerBase_hpp */
