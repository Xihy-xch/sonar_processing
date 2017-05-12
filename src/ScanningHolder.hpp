#ifndef ScanningHolder_hpp
#define ScanningHolder_hpp

#include <stdio.h>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Rock includes
#include <base/Angle.hpp>
#include <base/samples/Sonar.hpp>

namespace sonar_processing {

class ScanningHolder {

public:
    ScanningHolder( uint width,
                    uint height)
                    : transfer_()
                    , accum_data_()
                    , motor_step_(base::Angle::fromRad(0))
                    , last_diff_step_(base::Angle::fromRad(0))
                    , num_steps_(0)
                    , last_sonar_()
    {
        cart_image_ = cv::Mat::zeros(width, height, CV_32F);
        cart_mask_  = cv::Mat::zeros(width, height, CV_8U);
        left_limit_ = base::Angle::Min();
        right_limit_ = base::Angle::Max();
    };

    ScanningHolder( uint width,
                    uint height,
                    base::Angle left_limit,
                    base::Angle right_limit)
                    : transfer_()
                    , accum_data_()
                    , left_limit_(left_limit)
                    , right_limit_(right_limit)
                    , motor_step_(base::Angle::fromRad(0))
                    , last_diff_step_(base::Angle::fromRad(0))
                    , num_steps_(0)
                    , last_sonar_()
    {
        cart_image_ = cv::Mat::zeros(width, height, CV_32F);
        cart_mask_  = cv::Mat::zeros(width, height, CV_8U);
    };

    ~ScanningHolder(){};

    /**
     * Updates the sonar view and local occupancy grid with the current sonar frame.
     * @param sonar - the sonar data
     */
    void update(const base::samples::Sonar& sonar);

    /**
     * Plot the current sonar view as a cv::Mat.
     * @return the sonar view
     */
    void drawSonarData();

    /**
     * Return the cartesian representation of current sonar data
     * @return the cartesian image
     */
    cv::Mat getCartImage() const {
        return cart_image_;
    }

    /**
     * Return the cartesian representation of current sonar data
     * @return the cartesian image
     */
    cv::Mat getCartMask() const {
        return cart_mask_;
    }

    /**
     * Define the sonar scanning sector.
     */
    void setSectorScan(base::Angle left_limit, base::Angle right_limit) {
        left_limit_ = left_limit;
        right_limit_ = right_limit;
    }

private:
    /* the transfer vector between image pixels and sonar data */
    std::vector<int> transfer_;

    /* the accumulated data to be display on sonar view */
    std::vector<float> accum_data_;

    /* the view of accumulated sonar readings */
    cv::Mat cart_image_;

    /* the drawable area */
    cv::Mat cart_mask_;

    /* the sector scan */
    base::Angle left_limit_, right_limit_;

    /* the angular difference between two beams and the previous one */
    base::Angle motor_step_, last_diff_step_;

    /* the number of possible beams by the motor_step */
    int num_steps_;

    /* the previous sonar reading */
    base::samples::Sonar last_sonar_;

protected:
    /**
    * Checks if the motor step angle size changed.
    * @param bearing - the current beam angle
    * @return true or false
    */
    bool isMotorStepChanged(const base::Angle& bearing);

    /**
    * Correlates each pixel image with its corresponding sonar data.
    * @param sonar - the sonar data
    */
    void generateTransferTable(const base::samples::Sonar& sonar);
};

} /* namespace sonar_processing */

#endif /* sonar_processing_ScanningHolder_hpp */
