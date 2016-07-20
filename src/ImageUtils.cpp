#include "sonar_target_tracking/ImageUtils.hpp"

namespace sonar_target_tracking {

void image_utils::cv32f_equalize_histogram(cv::Mat src, cv::Mat dst) {
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);
    cv::equalizeHist(aux, aux);
    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

void image_utils::cv32f_clahe(cv::Mat src, cv::Mat dst, double clip_size, cv::Size grid_size) {
    cv::Mat aux;
    src.convertTo(aux, CV_8UC1, 255);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clip_size, grid_size);
    clahe->apply(aux, aux);
    
    aux.convertTo(dst, CV_32F,  1.0 / 255);
}

} /* sonar_target_tracking image_utils */
     
