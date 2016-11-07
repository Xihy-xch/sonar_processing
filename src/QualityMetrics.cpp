#include "QualityMetrics.hpp"

namespace sonar_processing {

double qs::MSE(cv::Mat& I1, cv::Mat& I2) {
    CV_Assert(I1.depth() == CV_32F && I2.depth() == CV_32F);

    cv::Mat diff;
    cv::absdiff(I1, I2, diff);
    diff = diff.mul(diff);

    double mse = cv::sum(diff)[0] / (I1.total());
    return mse;
}

double qs::RMSE(cv::Mat& I1, cv::Mat& I2) {
    return sqrt(MSE(I1, I2));
}

double qs::PSNR(cv::Mat& I1, cv::Mat& I2) {
    CV_Assert(I1.depth() == CV_32F && I2.depth() == CV_32F);

    double mse = MSE(I1,  I2);
    double psnr = 10.0 * log10((255 * 255) / mse);
    return psnr;
}

cv::Scalar qs::MSSIM( const cv::Mat& I1, const cv::Mat& I2) {
    CV_Assert(I1.depth() == CV_32F && I2.depth() == CV_32F);

    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    I1.convertTo(I1, CV_32F);           // cannot calculate on one byte large values
    I2.convertTo(I2, CV_32F);

    cv::Mat I1_2   = I1.mul(I1);        // I1^2
    cv::Mat I2_2   = I2.mul(I2);        // I2^2
    cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

    /*************************** END INITS **********************************/

    cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2   =   mu1.mul(mu1);
    cv::Mat mu2_2   =   mu2.mul(mu2);
    cv::Mat mu1_mu2 =   mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    cv::Mat t1, t2, t3;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map
    return mssim;
}

} /* sonar_processing image_util */
