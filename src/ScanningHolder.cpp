#include "ScanningHolder.hpp"

namespace sonar_processing {

// update the obstacle detection with current sonar frame
void ScanningHolder::update(const base::samples::Sonar& sonar) {
    if (!sonar.bin_count || !(sonar.beam_count == 1))
        return;

    // check if the motor step size is changed
    bool changedMotorStep = last_sonar_.beam_count && isMotorStepChanged(sonar.bearings[0]);

    if ((changedMotorStep || sonar.bin_count != last_sonar_.bin_count) && motor_step_.rad && last_sonar_.beam_count) {

        // set the transfer vector between image pixels and sonar data
        generateTransferTable(sonar);

        // resets the accumulated sonar data
        if (changedMotorStep || sonar.bin_count != last_sonar_.bin_count)
            accum_data_.assign(num_steps_ * sonar.bin_count, 0.0);
    }

    // add the current sonar data
    if (accum_data_.size()) {
        int id_beam = round((num_steps_ - 1) * (sonar.bearings[0].rad + M_PI) / (2 * M_PI));

        for (size_t i = 0; i < sonar.bin_count; ++i)
            accum_data_[id_beam * sonar.bin_count + i] = sonar.bins[i];
    }

    drawSonarData();
    last_sonar_ = sonar;
}

// draw the sector scanning data
void ScanningHolder::drawSonarData() {
    for (size_t i = 0; i < transfer_.size(); ++i) {
        if (transfer_[i] != -1) {
            cart_image_.at<float>(i / cart_image_.cols, i % cart_image_.cols) = accum_data_[transfer_[i]];
        }
    }
}

// check is the motor step angle size is changed
bool ScanningHolder::isMotorStepChanged(const base::Angle& bearing) {
    base::Angle diff_step = bearing - last_sonar_.bearings[0];
    diff_step.rad = fabs(diff_step.rad);

    // if the sector scanning is enabled, the diffStep could be lower than motorStep when the bearing is closer to one of the corners
    if (fabs((left_limit_ - bearing).rad) < motor_step_.rad || fabs((right_limit_ - bearing).rad) < motor_step_.rad) {
        last_diff_step_ = diff_step;
        return false;
    }

    if (!motor_step_.isApprox(diff_step) && last_diff_step_.isApprox(diff_step)) {
        motor_step_ = diff_step;
        num_steps_ = M_PI * 2 / motor_step_.rad;
        last_diff_step_ = diff_step;
        return true;
    }

    last_diff_step_ = diff_step;
    return false;
}

// set the transfer vector between image pixels and sonar data
void ScanningHolder::generateTransferTable(const base::samples::Sonar& sonar) {
    transfer_.clear();
    cart_image_.setTo(0);
    cart_mask_.setTo(0);

    if (!motor_step_.rad)
        return;

    // set the origin
    cv::Point2f origin(cart_image_.cols / 2, cart_image_.rows / 2);

    for (size_t j = 0; j < cart_image_.rows; j++) {
        for (size_t i = 0; i < cart_image_.cols; i++) {
            // current point
            cv::Point2f point(i - origin.x, j - origin.y);
            point.x = point.x * (float) sonar.bin_count / (cart_image_.cols * 0.5);
            point.y = point.y * (float) sonar.bin_count / (cart_image_.rows * 0.5);

            double radius = sqrt(point.x * point.x + point.y * point.y);
            double angle = atan2(-point.x, -point.y);

            // pixels out the sonar image
            if(radius > sonar.bin_count || !radius || angle < left_limit_.rad || angle > right_limit_.rad)
                transfer_.push_back(-1);

            // pixels in the sonar image
            else {
                cart_mask_.at<uchar>(j,i) = 255;
                int id_beam = round((num_steps_ - 1) * (angle + M_PI) / (2 * M_PI));
                transfer_.push_back(id_beam * sonar.bin_count + radius);
            }
        }
    }
}
} /* namespace sonar_processing */
