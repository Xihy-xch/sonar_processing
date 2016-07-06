#include "sonar_target_tracking/TargetTrack.hpp"

namespace sonar_target_tracking 
{

TargetTrack::TargetTrack(
    const std::vector<float>& bins,
    const std::vector<float>& bearings,
    uint32_t beam_count,
    uint32_t bin_count)
    : bins_(bins)
    , bearings_(bearings)
    , beam_count_(beam_count)
    , bin_count_(bin_count) {

}

}
