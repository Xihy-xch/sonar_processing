#ifndef sonar_processing_HogDescriptorViz_hpp
#define sonar_processing_HogDescriptorViz_hpp

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size& size);

#endif
