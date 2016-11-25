#ifndef Clustering_hpp
#define Clustering_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <deque>

namespace sonar_processing {

namespace clustering {

void kmeans(const cv::Mat& input, cv::Mat& output, int cluster_number);

} /* namespace clustering */

} /* namespace sonar_processing */

#endif /* sonar_processing_Clustering_hpp */
