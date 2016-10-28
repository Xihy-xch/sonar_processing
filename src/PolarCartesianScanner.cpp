#include "PolarCartesianScanner.hpp"

namespace sonar_processing {

PolarCartesianScanner::PolarCartesianScanner(const SonarHolder& sonar_holder)
    : sonar_holder_(sonar_holder)
{
}

PolarCartesianScanner::~PolarCartesianScanner() {
}

void PolarCartesianScanner::GetCartesianLine(int from_index, std::vector<int>& indices, int total_elements) {
    GetCartesianLine(from_index, cv::Point2f(-1, -1), indices, total_elements);
}

void PolarCartesianScanner::GetCartesianLine(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements) {
    int total_elements_2 = (total_elements == -1) ? -1 : total_elements / 2;
    PerformCartesianScanning(indices, left_scanning_arguments(from_index, from_point), total_elements_2);
    std::reverse(indices.begin(), indices.end());
    indices.push_back(from_index);
    PerformCartesianScanning(indices, right_scanning_arguments(from_index, from_point), total_elements_2);
}

void PolarCartesianScanner::GetCartesianColumn(int from_index, std::vector<int>& indices, int total_elements) {
    GetCartesianColumn(from_index, cv::Point2f(-1, -1), indices, total_elements);
}

void PolarCartesianScanner::GetCartesianColumn(int from_index, cv::Point2f from_point, std::vector<int>& indices, int total_elements) {
    int total_elements_2 = (total_elements == -1) ? -1 : total_elements / 2;
    indices.clear();
    PerformCartesianScanning(indices, up_scanning_arguments(from_index, from_point), total_elements_2);
    std::reverse(indices.begin(), indices.end());
    indices.push_back(from_index);
    PerformCartesianScanning(indices, down_scanning_arguments(from_index, from_point), total_elements_2);
}

void PolarCartesianScanner::GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices) {
    indices.assign(nsize * nsize, -1);

    std::vector<int> line_indices;
    std::vector<cv::Point2f> line_points;
    std::vector<int> transpose_indices(nsize * nsize, -1);
    
    GetIntersectionLine(from_index, nsize, line_indices, line_points);

    for (size_t col = 0; col < line_indices.size(); col++) {
        std::vector<int> column_indices;
        GetCartesianColumn(line_indices[col], line_points[col], column_indices, nsize);
        std::copy(column_indices.begin(), column_indices.end(), transpose_indices.begin() + col * nsize);
    }
    
    for (size_t col = 0; col < nsize; col++) {
        for (size_t line = 0; line < nsize; line++) {
            indices[line * nsize + col] = transpose_indices[col * nsize + line];
        }
    }
}

void PolarCartesianScanner::GetCartesianNeighborhood(int from_index, int nsize, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    indices.assign(nsize * nsize, -1);
    points.assign(nsize * nsize, cv::Point2f(-1, -1));

    std::vector<int> line_indices;
    std::vector<cv::Point2f> line_points;
    std::vector<int> transpose_indices(nsize * nsize, -1);
    std::vector<cv::Point2f> transpose_points(nsize * nsize, cv::Point2f(-1, -1));

    GetIntersectionLine(from_index, nsize, line_indices, line_points);

    for (size_t col = 0; col < line_indices.size(); col++) {
        std::vector<int> column_indices;
        std::vector<cv::Point2f> column_points;

        GetCartesianColumn(line_indices[col], line_points[col], column_indices, nsize);
        EvaluateIntersectionPoints(line_indices[col], line_points[col], column_indices, kScanningVertical, column_points);

        std::copy(column_indices.begin(), column_indices.end(), transpose_indices.begin() + col * nsize);
        std::copy(column_points.begin(), column_points.end(), transpose_points.begin() + col * nsize);
    }

    for (size_t col = 0; col < nsize; col++) {
        for (size_t line = 0; line < nsize; line++) {
            indices[line * nsize + col] = transpose_indices[col * nsize + line];
            points[line * nsize + col] = transpose_points[col * nsize + line];
        }
    }
}

void PolarCartesianScanner::GetIntersectionLine(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    GetCartesianLine(from_index, indices, total_elements);
    EvaluateIntersectionPoints(from_index, sonar_holder_.cart_center_point(from_index), indices, kScanningHorizontal, points);
}

void PolarCartesianScanner::GetIntersectionColumn(int from_index, int total_elements, std::vector<int>& indices, std::vector<cv::Point2f>& points) {
    GetCartesianColumn(from_index, indices, total_elements);
    EvaluateIntersectionPoints(from_index, sonar_holder_.cart_center_point(from_index), indices, kScanningVertical, points);
}

void PolarCartesianScanner::PerformCartesianScanning(std::vector<int>& indices, ScanningArguments args, int total_elements) {

    int count = 0;
    int index = args.from_index;

    cv::Point2f from_point;

    if (args.from_point.x == -1 || args.from_point.y == -1) {
        args.from_point = sonar_holder_.cart_center_point(args.from_index);
    }

    index = NextCartesianIndex(index, args);

    while (!IsScanningComplete(index, count, total_elements)) {
        indices.push_back(index);
        index = NextCartesianIndex(index, args);
        count++;
    }
}

int PolarCartesianScanner::NextCartesianIndex(int index, ScanningArguments args) {
    std::vector<int> indices;
    sonar_holder_.GetNeighborhood(index, indices);

    int minimum_distance_index = GetMinimumDistanceIndex(index, indices, args);
    return indices[minimum_distance_index];
}

int PolarCartesianScanner::GetMinimumDistanceIndex(int index, const std::vector<int>& indices, ScanningArguments args) {
    int minimum_distance_index = -1;
    float minimum_distance = FLT_MAX;
    float neighbor_distance = 0;

    cv::Point2f current_point = sonar_holder_.cart_center_point(index);

    for (size_t i = 0; i < indices.size(); i++) {

        if (indices[i] != -1) {
            cv::Point2f neighbor_point = sonar_holder_.cart_center_point(indices[i]);

            if (call_compare_function(args.compare_function, neighbor_point, current_point)) {
                neighbor_distance = call_distance_function(args.distance_function, args.from_point, neighbor_point);
                if (neighbor_distance < minimum_distance) {
                    minimum_distance = neighbor_distance;
                    minimum_distance_index = i;
                }
            }
        }
    }

    return minimum_distance_index;
}

bool PolarCartesianScanner::IsScanningComplete(int index, int count, int total_elements) {
    int beam = sonar_holder_.index_to_beam(index);
    int bin = sonar_holder_.index_to_bin(index);

    if (beam <= 0 || beam >= sonar_holder_.beam_count()-2 ||
        bin <= 0 || bin >= sonar_holder_.bin_count()-2 ||
        (total_elements != -1 && count >= total_elements) ||
        index == -1) {
        return true;
    }

    return false;
}

void PolarCartesianScanner::EvaluateIntersectionPoints(int from_index, cv::Point2f from_point, const std::vector<int>& indices, ScanningDirection direction, std::vector<cv::Point2f>& points) {
    points.assign(indices.size(), cv::Point2f(-1, -1));

    for (size_t i = 0; i < indices.size(); i++) {
        cv::Point2f start_point, final_point;
        EvaluateSectorLimitPoints(indices[i], from_point, direction, start_point, final_point);
        SetMiddlePoint(direction, start_point, final_point, points[i]);
    }
}

void PolarCartesianScanner::EvaluateSectorLimitPoints (
    int from_index,
    cv::Point2f from_point,
    ScanningDirection direction,
    cv::Point2f& start_point,
    cv::Point2f& final_point) {

    cv::Point2f origin = sonar_holder_.cart_origin();
    cv::Rect_<float> rc = sonar_holder_.sector_bounding_rect(from_index);

    float start_radius, final_radius;
    float start_theta, final_theta;
    
    sonar_holder_.GetPolarLimits(from_index, start_radius, final_radius, start_theta, final_theta);

    start_point = cv::Point2f(-1, -1);
    final_point = cv::Point2f(-1, -1);

    float start_position, final_position, scanning_position;
    
    float resolution = 0;
    if (direction == kScanningHorizontal) {
        resolution = rc.width / 10.0;
    }
    else if (direction == kScanningVertical) {
        resolution = rc.height / 10.0;
    }

    SetScanningPosition(rc, from_point, resolution, direction, start_position, final_position, scanning_position);

    bool inside_sector = false;
    for (float position=start_position; position<=final_position; position+=resolution) {
        cv::Point2f pt = scanning_point(direction, position, scanning_position);

        float dx = origin.x - pt.x;
        float dy = origin.y - pt.y;
        float r = sqrt(dx * dx + dy * dy);
        float t = atan2f(dy, dx) - M_PI_2;

        if (r >= start_radius && r <= final_radius &&
            t >= start_theta && t <= final_theta) {
            if (start_point.x == -1 && start_point.y == -1) start_point = pt;
            inside_sector = true;
        }
        else {
            if (inside_sector) final_point = pt;
            inside_sector = false;
        }
    }
}

float PolarCartesianScanner::GetMinimumRadius(const std::vector<int>& indices) {
    int minimum_bin = INT_MAX;
    int bin = 0;

    for (size_t i = 0; i < indices.size(); i++) {
        bin = sonar_holder_.index_to_bin(indices[i]);
        if (bin < minimum_bin) minimum_bin = bin;
    }

    return minimum_bin;
}

void PolarCartesianScanner::SetScanningPosition(
    cv::Rect_<float> rc,
    cv::Point2f from_point,
    float resolution,
    ScanningDirection direction,
    float& start_position,
    float& final_position,
    float& scanning_position) {

    if (direction == kScanningHorizontal) {
        start_position = rc.tl().x;
        final_position = rc.br().x+resolution;
        scanning_position = from_point.y;
    }
    else if (direction == kScanningVertical){
        start_position = rc.tl().y;
        final_position = rc.br().y+resolution;
        scanning_position = from_point.x;
    }
    else {
        throw std::invalid_argument("Invalid scanning direction.");
    }
}

void PolarCartesianScanner::SetMiddlePoint(ScanningDirection d, cv::Point2f start_point, cv::Point2f final_point, cv::Point2f& result_point) {
    if (d == kScanningHorizontal) {
        result_point = cv::Point2f(start_point.x + (final_point.x - start_point.x) / 2, start_point.y);
    }
    else if (d == kScanningVertical) {
        result_point = cv::Point2f(start_point.x, start_point.y + (final_point.y - start_point.y) / 2);
    }
    else {
        throw std::invalid_argument("Invalid scanning direction.");
    }
}

}
