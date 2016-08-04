#ifndef TargetTrack_hpp
#define TargetTrack_hpp

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sonar_target_tracking {

class TargetTrack {
public:
    TargetTrack(const std::vector<float>& bins,
                const std::vector<float>& bearings,
                uint32_t beam_count,
                uint32_t bin_count);

    ~TargetTrack() {}

    void apply();

    void remove_background(std::vector<double> features, uint32_t bsize);

    cv::Rect horiz_roi() const {
        return horiz_roi_;
    }

private:

    const std::vector<float>& bins_;
    const std::vector<float>& bearings_;

    uint32_t beam_count_;
    uint32_t bin_count_;

    cv::Rect horiz_roi_;
};

}

#endif /* end of include guard: TargetTrack_hpp */
