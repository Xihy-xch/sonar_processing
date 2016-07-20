#ifndef ImageUtils_hpp
#define ImageUtils_hpp

#include <iostream>
#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {
    
namespace image_utils {
    
template <typename T>
std::vector<T> mat2vector(cv::Mat mat) {
    std::vector<T> array;
    if (mat.isContinuous()) {
        array.assign((T*)mat.datastart, (T*)mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), (T*)mat.ptr<uchar>(i), (T*)mat.ptr<uchar>(i)+mat.cols);
        }
    }
    return array;    
}

void cv32f_equalize_histogram(cv::Mat src, cv::Mat dst);

void cv32f_clahe(cv::Mat src, cv::Mat dst, double clip_size = 40.0, cv::Size grid_size = cv::Size(8, 8));


} /* namespace image_utils */
 
} /* sonar_target_tracking image_utils */

#endif /* ImageUtils_hpp */
