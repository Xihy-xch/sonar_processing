#ifndef sonar_processing_PolarCartesianScanner_hpp
#define sonar_processing_PolarCartesianScanner_hpp

#include <stdio.h>
#include "sonar_processing/SonarHolder.hpp"

namespace sonar_processing {

class PolarCartesianScanner {

public:
    PolarCartesianScanner(const SonarHolder& sonar_holder);
    ~PolarCartesianScanner();

    void GetCartesianLine(int from_index, std::vector<int>& indices, int total_elements = -1);

    void GetCartesianLine(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements = -1);

    void GetCartesianColumn(int from_index, std::vector<int>& indices, int total_elements = -1);

    void GetCartesianColumn(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements = -1);

    void GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices);

    void GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices, std::vector<cv::Point2f>& points);

    void GetIntersectionLine(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points);

    void GetIntersectionColumn(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points);

private:

    enum ScanningDirection {
        kScanningHorizontal = 0,
        kScanningVertical
    };

    typedef float (PolarCartesianScanner::*DistanceFunction)(cv::Point2f pt0, cv::Point2f pt1);
    typedef bool  (PolarCartesianScanner::*CompareFunction )(cv::Point2f pt0, cv::Point2f pt1);

    struct ScanningArguments {

        ScanningArguments(int from_index, cv::Point2f from_point, DistanceFunction distance_function, CompareFunction compare_function)
            : from_index(from_index)
            , from_point(from_point)
            , distance_function(distance_function)
            , compare_function(compare_function)
        {
        }

        ScanningArguments() {
            from_index = -1;
            from_point = cv::Point2f(-1, -1);
            distance_function = NULL;
            compare_function = NULL;
        }

        int from_index;

        cv::Point2f from_point;
        DistanceFunction distance_function;
        CompareFunction compare_function;
    };

    float call_distance_function(DistanceFunction distance_function, cv::Point2f pt0, cv::Point2f pt1) {
        return (this->*distance_function)(pt0, pt1);
    }

    bool call_compare_function(CompareFunction compare_function, cv::Point2f pt0, cv::Point2f pt1) {
        return (this->*compare_function)(pt0, pt1);
    }

    float horizontal_distance(cv::Point2f pt0, cv::Point2f pt1) {
        return fabs(pt0.y - pt1.y);
    }

    float vertical_distance(cv::Point2f pt0, cv::Point2f pt1) { return fabs(pt0.x - pt1.x); }

    bool left_compare(cv::Point2f pt0, cv::Point2f pt1) {
        return pt0.x < pt1.x;
    }

    bool right_compare(cv::Point2f pt0, cv::Point2f pt1) {
        return pt0.x > pt1.x;
    }

    bool up_compare(cv::Point2f pt0, cv::Point2f pt1) { return pt0.y < pt1.y; }

    bool down_compare(cv::Point2f pt0, cv::Point2f pt1) { return pt0.y > pt1.y; }

    ScanningArguments horizontal_scanning_arguments(int from_index, cv::Point2f from_point) {
        return ScanningArguments(from_index, from_point, &PolarCartesianScanner::horizontal_distance, NULL);
    }

    ScanningArguments left_scanning_arguments(int from_index, cv::Point2f from_point) {
        ScanningArguments args = horizontal_scanning_arguments(from_index, from_point);
        args.compare_function = &PolarCartesianScanner::left_compare;
        return args;
    }

    ScanningArguments right_scanning_arguments(int from_index, cv::Point2f from_point) {
        ScanningArguments args = horizontal_scanning_arguments(from_index, from_point);
        args.compare_function = &PolarCartesianScanner::right_compare;
        return args;
    }

    ScanningArguments vertical_scanning_arguments(int from_index, cv::Point2f from_point) {
        return ScanningArguments(from_index, from_point, &PolarCartesianScanner::vertical_distance, NULL);
    }

    ScanningArguments up_scanning_arguments(int from_index, cv::Point2f from_point) {
        ScanningArguments args = vertical_scanning_arguments(from_index, from_point);
        args.compare_function = &PolarCartesianScanner::up_compare;
        return args;
    }

    ScanningArguments down_scanning_arguments(int from_index, cv::Point2f from_point) {
        ScanningArguments args = vertical_scanning_arguments(from_index, from_point);
        args.compare_function = &PolarCartesianScanner::down_compare;
        return args;
    }

    void PerformCartesianScanning(std::vector<int>& indices, ScanningArguments args, int total_elements = -1);

    int NextCartesianIndex(int index, ScanningArguments args);

    int GetMinimumDistanceIndex(int index, const std::vector<int>& indices, ScanningArguments args);

    bool IsScanningComplete(int index, int count, int total_elements = -1);

    void EvaluateIntersectionPoints(int from_index, cv::Point2f from_point, const std::vector<int>& indices, ScanningDirection direction, std::vector<cv::Point2f>& points);

    void EvaluateSectorLimitPoints(
        int from_index,
        cv::Point2f from_point,
        ScanningDirection direction,
        cv::Point2f& start_point,
        cv::Point2f& final_point);

    float GetMinimumRadius(const std::vector<int>& indices);

    void SetScanningPosition (
        cv::Rect_<float> rc,
        cv::Point2f from_point,
        float resolution,
        ScanningDirection direction,
        float& start_position,
        float& final_position,
        float& scanning_position);

    cv::Point2f scanning_point(ScanningDirection d, float x, float y) {
        return (d == kScanningHorizontal) ? cv::Point2f(x, y) :
               ((d == kScanningVertical)  ? cv::Point2f(y, x) :
               cv::Point2f(-1. -1));
    }

    void SetMiddlePoint(ScanningDirection d, cv::Point2f start_point, cv::Point2f final_point, cv::Point2f& result_point);

    const SonarHolder& sonar_holder_;

};




} // sonar_processing


#endif /* PolarCartesianScanner_hpp */
