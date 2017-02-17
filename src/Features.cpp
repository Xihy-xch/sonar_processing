#include "sonar_processing/Features.hpp"

#include <Eigen/Eigenvalues>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

namespace sonar_processing {

namespace features {

void calculateCovarianceMatrix(std::vector<std::vector<cv::Point> > contours, cv::Mat reference_image, std::vector<cv::Mat>& covar_matrices) {
    cv::Mat dx, dy, dxx, dyy;
    // first and second derivatives (horizontally and vertically)
    cv::Sobel(reference_image,  dx, CV_32F, 1, 0, 3);
    cv::Sobel(reference_image,  dy, CV_32F, 0, 1, 3);
    cv::Sobel(reference_image, dxx, CV_32F, 2, 0, 3);
    cv::Sobel(reference_image, dyy, CV_32F, 0, 2, 3);

    // magnitude and direction of gradients
    cv::Mat magnitude, direction;
    cv::magnitude(dx, dy, magnitude);
    cv::phase(dx, dy, direction, false);

    for (size_t i = 0; i < contours.size(); i++) {          // all blobs
        cv::Mat blob_description = cv::Mat::zeros(cv::Size(9, contours[i].size()), CV_32FC1);
        for (size_t j = 0; j < contours[i].size(); j++) {   // all pixels of same blob
            // positions
            blob_description.at<float>(j, 0) = contours[i][j].x;
            blob_description.at<float>(j, 1) = contours[i][j].y;
            // intensity
            blob_description.at<float>(j, 2) = reference_image.at<float>(contours[i][j].y, contours[i][j].x);
            // first and second derivatives
            blob_description.at<float>(j, 3) =  dx.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 4) =  dy.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 5) = dxx.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 6) = dyy.at<float>(contours[i][j].y, contours[i][j].x);
            // gradient magnitude and directions
            blob_description.at<float>(j, 7) = magnitude.at<float>(contours[i][j].y, contours[i][j].x);
            blob_description.at<float>(j, 8) = direction.at<float>(contours[i][j].y, contours[i][j].x);
        }
        // calculate covariance matrix
        cv::Mat covar, mean;
        cv::calcCovarMatrix(blob_description, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        covar = covar / (blob_description.rows - 1);
        covar_matrices.push_back(covar);
    }
}

double riemannianDistance(cv::Mat A, cv::Mat B) {
    Eigen::MatrixXf e_A, e_B;
    cv::cv2eigen(A, e_A);
    cv::cv2eigen(B, e_B);

    /* compute generalized eigenvalues */
    Eigen::GeneralizedEigenSolver<Eigen::MatrixXf> ges;
    ges.compute(e_A, e_B);
    cv::Mat gev;
    cv::eigen2cv(Eigen::MatrixXf(ges.eigenvalues().real()), gev);

    /* riemannian distance */
    cv::Mat partial_res;
    cv::log(gev, partial_res);
    cv::pow(partial_res, 2, partial_res);
    double distance = sqrt(cv::sum(partial_res)[0]);
    return distance;
}

} /* namespace features */

} /* namespace sonar_processing */
