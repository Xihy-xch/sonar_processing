#ifndef sonar_target_tracking_BasicOperations_hpp
#define sonar_target_tracking_BasicOperations_hpp

#include "sonar_target_tracking/SonarHolder.hpp"
#include "sonar_target_tracking/PolarCartesianScanner.hpp"

namespace sonar_target_tracking {

namespace basic_operations {

inline void line_indices(const SonarHolder& sonar_holder, int from_index, std::vector<int>& indices) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetCartesianLine(from_index, indices);
}

inline void line_indices_from_bin(const SonarHolder& sonar_holder, int bin, std::vector<int>& indices) {
    line_indices(sonar_holder, sonar_holder.index_at(sonar_holder.beam_count() / 2, bin), indices);
}

inline void column_indices(const SonarHolder& sonar_holder, int from_index, std::vector<int>& indices) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetCartesianColumn(from_index, indices);
}

inline void column_indices_from_bin(const SonarHolder& sonar_holder, int bin, std::vector<int>& indices) {
    column_indices(sonar_holder, sonar_holder.index_at(sonar_holder.beam_count() / 2, bin), indices);
}

inline void line_values(const SonarHolder& sonar_holder, int from_index, std::vector<float>& values) {
    std::vector<int> indices;
    line_indices(sonar_holder, from_index, indices);
    sonar_holder.values(indices, values);
}

inline void column_values(const SonarHolder& sonar_holder, int from_index, std::vector<float>& values) {
    std::vector<int> indices;
    column_indices(sonar_holder, from_index, indices);
    sonar_holder.values(indices, values);
}

inline void intersetion_line(const SonarHolder& sonar_holder, int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetIntersectionLine(from_index, total_elements, indices, points);
}

inline void intersetion_column(const SonarHolder& sonar_holder, int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetIntersectionColumn(from_index, total_elements, indices, points);
}

inline void neighborhood(const SonarHolder& sonar_holder, int from_index, int nsize, std::vector<int>& indices) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetCartesianNeighborhood(from_index, nsize, indices);
}

inline void neighborhood(const SonarHolder& sonar_holder, int from_index, int nsize, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    PolarCartesianScanner polar_cartesian_scanner(sonar_holder);
    polar_cartesian_scanner.GetCartesianNeighborhood(from_index, nsize, indices, points);
}

inline float line_sum(const SonarHolder& sonar_holder, int from_index) {
    std::vector<float> values;
    line_values(sonar_holder, from_index, values);
    return cv::sum(cv::Mat1f(values))[0];
}

inline float line_mean(const SonarHolder& sonar_holder, int from_index) {
    std::vector<float> values;
    line_values(sonar_holder, from_index, values);
    return cv::mean(cv::Mat1f(values))[0];
}

inline void average_lines(const SonarHolder& sonar_holder, int first_bin, int last_bin, std::vector<float>& avgs) {
    int beam = sonar_holder.beam_count() / 2;
    avgs.resize(last_bin-first_bin);
    for (int bin = first_bin, i = 0; bin < last_bin; bin++, i++) {
        avgs[i] = line_mean(sonar_holder, sonar_holder.index_at(beam, bin));
    }
}

} // namespace basic_operations

} // namespace sonar_target_tracking

#endif /* end of include guard: sonar_target_tracking_BasicOperations_hpp */
