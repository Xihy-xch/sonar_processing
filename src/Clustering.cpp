#include "sonar_processing/Clustering.hpp"

namespace sonar_processing {

namespace clustering {

void kmeans (const cv::Mat& src, cv::Mat& dst, int cluster_number) {
    CV_Assert(src.depth() == CV_32FC1);

    cv::Mat reshaped = src.reshape(1, src.cols * src.rows);

    // run kmeans
    cv::Mat labels, centers;
    int attempts = 4;
    cv::kmeans(reshaped, cluster_number, labels, cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), attempts, cv::KMEANS_RANDOM_CENTERS, centers);

    // results
    dst.create(src.rows, src.cols, CV_32F);
    for (size_t y = 0; y < dst.rows; y++) {
        for (size_t x = 0; x < dst.cols; x++) {
            int cluster_idx = labels.at<int>(y * dst.cols + x, 0);
            dst.at<float>(y, x) = centers.at<float>(cluster_idx, 0);
        }
    }
}

} /* namespace clustering */

} /* namespace sonar_processing */
