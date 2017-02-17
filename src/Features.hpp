#ifndef Features_hpp
#define Features_hpp

#include <opencv2/opencv.hpp>

namespace sonar_processing {

namespace features {

void calculateCovarianceMatrix(std::vector<std::vector<cv::Point> > contours, cv::Mat reference_image, std::vector<cv::Mat>& covar_matrices);

double riemannianDistance(cv::Mat A, cv::Mat B);

} /* namespace denoising */

} /* namespace sonar_processing */

#endif /* sonar_processing_Features_hpp */
