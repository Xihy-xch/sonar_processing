#include "ShapeProcessing.hpp"

namespace sonar_processing {

namespace shape_processing {

std::vector<std::vector<cv::Point> > convexhull(std::vector<std::vector<cv::Point> > contours) {
    std::vector<std::vector<cv::Point> >hull(contours.size());

    for( uint32_t i = 0; i < contours.size(); i++ ) {
        cv::convexHull( cv::Mat(contours[i]), hull[i], false );
    }

    return hull;
}


std::vector<std::vector<cv::Point> > find_contours(cv::InputArray src_arr, int mode, bool convex_hull) {
    std::vector<std::vector<cv::Point> > contours;

    cv::findContours(src_arr.getMat(), contours, mode, CV_CHAIN_APPROX_SIMPLE);

    if (convex_hull) return convexhull(contours);

    return contours;
}

std::vector<std::vector<cv::Point> > find_contours(cv::InputArray src_arr, cv::Size min_size, int mode, bool convex_hull) {
    std::vector<std::vector<cv::Point> > filtering_contours;
    std::vector<std::vector<cv::Point> > contours = find_contours(src_arr, mode, convex_hull);

    for( int i = 0; i < contours.size(); i++ ) {
        cv::Rect rc = cv::boundingRect( cv::Mat(contours[i]) );
        if (rc.width > min_size.width &&
            rc.height > min_size.height) {
            filtering_contours.push_back(contours[i]);
        }
    }

    return filtering_contours;
}

} /* namespace shape_processing */

} /* namespace sonar_processing */
