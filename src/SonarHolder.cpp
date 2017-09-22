#include <cstdio>
#include <algorithm>
#include "MathUtil.hpp"
#include "SonarHolder.hpp"

namespace sonar_processing {

SonarHolder::SonarHolder()
    : beam_width_(0.0)
    , bin_count_(0)
    , beam_count_(0)
    , cart_size_(-1, -1)
    , cart_origin_(0.0, 0.0)
    , total_elements_(0)
    , interpolation_type_(LINEAR)
{
}

SonarHolder::SonarHolder(
    std::vector<float> bins,
    float start_beam,
    float beam_width,
    uint32_t bin_count,
    uint32_t beam_count,
    cv::Size cart_size,
    int interpolation_type)
    : cart_size_(-1, -1)
    , cart_origin_(0.0, 0.0)
    , total_elements_(0)
{
    Reset(bins, start_beam, beam_width, bin_count, beam_count, cart_size, interpolation_type);
}

SonarHolder::SonarHolder(
    std::vector<float> bins,
    std::vector<float> bearings,
    float beam_width,
    uint32_t bin_count,
    uint32_t beam_count,
    cv::Size cart_size,
    int interpolation_type)
    : cart_size_(-1, -1)
    , cart_origin_(0.0, 0.0)
    , total_elements_(0)
{
    Reset(bins, bearings, beam_width, bin_count, beam_count, cart_size, interpolation_type);
}

SonarHolder::~SonarHolder() {
}

void SonarHolder::Reset(
    std::vector<float> bins,
    std::vector<float> bearings,
    float beam_width,
    uint32_t bin_count,
    uint32_t beam_count,
    cv::Size cart_size,
    int interpolation_type)
{
    bool is_initialize = (bin_count == bin_count_ && beam_count == beam_count_);

    bins_ = bins;
    bearings_ = bearings;
    bin_count_ = bin_count;
    beam_count_ = beam_count;
    beam_width_ = beam_width;
    interpolation_type_ = interpolation_type;

    raw_image_ = cv::Mat(bins_).reshape(1, beam_count_);

    if (!is_initialize) {
        cart_size_ = cart_size;
        Initialize();
    }

    InitializeCartesianImage(bins_, cart_image_);
}

void SonarHolder::Reset(
    std::vector<float> bins,
    float start_beam,
    float beam_width,
    uint32_t bin_count,
    uint32_t beam_count,
    cv::Size cart_size,
    int interpolation_type)
{
    Reset(bins, BuildBeamBearings(start_beam, beam_width, beam_count), beam_width, bin_count, beam_count, cart_size, interpolation_type);
}
void SonarHolder::ResetBins(const std::vector<float>& bins){
    Reset(bins, bearings_, beam_width_, bin_count_, beam_count_);
}

void  SonarHolder::CreateCartesianImageFromCvMat(cv::InputArray raw_image, cv::OutputArray cart_image) const {
    std::vector<float> bins;
    bins.assign((float*)raw_image.getMat().datastart, (float*)raw_image.getMat().dataend);
    CreateCartesianImage(bins, cart_image);
}

void SonarHolder::CreateCartesianImage(const std::vector<float>& bins, cv::OutputArray cart_image) const {
    InitializeCartesianImage(bins, cart_image);
}


std::vector<float> SonarHolder::BuildBeamBearings(float start_beam, float beam_width, uint32_t beam_count) {
    std::vector<float> bearings;
    bearings.resize(beam_count, 0);
    float interval = beam_width / beam_count;
    float angle = start_beam;
    for (uint32_t beam = 0; beam <= beam_count; beam++, angle += interval) bearings[beam] = angle;
    return bearings;
}

void SonarHolder::Initialize() {
    total_elements_ = bin_count_ * beam_count_;
    cart_size_ref_ = cv::Size(cos(beam_width_ - M_PI_2) * bin_count_ * 2.0, bin_count_);

    if (cart_size_.width == -1 || cart_size_.height == -1) {
        cart_size_ = cart_size_ref_;
        cart_width_factor_ = 1.0;
        cart_height_factor_ = 1.0;
    }
    else {
        cart_width_factor_ = cart_size_.width / (float)cart_size_ref_.width;
        cart_height_factor_ = cart_size_.height / (float)cart_size_ref_.height;
    }

    cart_origin_ = cv::Point2f(cart_size_.width / 2, cart_size_.height - 1);
    bins_mask_.assign(total_elements_, 1);

    InitializeCartesianPoints();
    InitializePolarMapping();
    InitializeCartesianLineLimits();
    InitializeCartesianImageMask();
}

void SonarHolder::InitializeCartesianPoints() {
    cart_points_.assign(total_elements_, cv::Point2f(-1, -1));

    if (cart_size_ != cart_size_ref_) {
        cv::Point2f cart_origin_ref = cv::Point2f(cart_size_ref_.width / 2, cart_size_ref_.height - 1);
        for (uint32_t bin = 0; bin < bin_count_; bin++) {
            for (uint32_t beam = 0; beam < beam_count_; beam++) {
                float radius = (bin == 0) ? 0.0001 : (float)bin;
                cv::Point2f cart_point = base::MathUtil::to_cartesianf(bearings_[beam], radius, -M_PI_2) + cart_origin_ref;
                cart_point.x = cart_point.x * cart_width_factor_;
                cart_point.y = cart_point.y * cart_height_factor_;
                cart_points_[beam * bin_count_ + bin] = cart_point;
            }
        }
    }
    else {
        for (uint32_t bin = 0; bin < bin_count_; bin++) {
            for (uint32_t beam = 0; beam < beam_count_; beam++) {
                float radius = (bin == 0) ? 0.0001 : (float)bin;
                cart_points_[beam * bin_count_ + bin] = base::MathUtil::to_cartesianf(bearings_[beam], radius, -M_PI_2) + cart_origin_;
            }
        }
    }
}

void SonarHolder::InitializePolarMapping() {
    cart_center_points_.assign(total_elements_, cv::Point2f(-1, -1));
    cart_to_polar_.assign(cart_size_.width * cart_size_.height, -1);
    radius_.assign(cart_size_.width * cart_size_.height, 0);
    angles_.assign(cart_size_.width * cart_size_.height, 0);

    for (uint32_t i = 0; i < total_elements_; i++) {
        SetCartesianToPolarSector(i);
    }
}

void SonarHolder::InitializeCartesianLineLimits() {
    cart_line_limits_.assign(cart_size_.height * 2, -1);
    for (size_t y = 0; y < cart_size_.height; y++) {
        int mid_column = cart_size_.width / 2;

        for (int x = mid_column; x >= 0 && cart_to_polar_index(x, y) != -1; x--) {
            cart_line_limits_[y * 2 + 0] = x;
        }

        for (int x = mid_column; x < cart_size_.width && cart_to_polar_index(x, y) != -1; x++) {
            cart_line_limits_[y * 2 + 1] = x;
        }
    }
}

void SonarHolder::InitializeCartesianImage(const std::vector<float>& bins, cv::OutputArray dst) const {
    if (interpolation_type_ == LINEAR) {
        LinearPolarToCartesianImage(bins, dst);
    }
    else if (interpolation_type_ == WEIGHTED)  {
        WeightedPolarToCartesianImage(bins, dst);
    }
    else {
        throw std::invalid_argument("the interpolation type is invalid");
    }
}

void SonarHolder::InitializeCartesianImageMask() {
    cart_image_mask_ = cv::Mat::zeros(cart_size_, CV_8UC1);
    uchar *ptr = reinterpret_cast<uchar*>(cart_image_mask_.data);
    for (size_t cart_idx = 0; cart_idx < cart_to_polar_.size(); cart_idx++) {
        if (cart_to_polar_[cart_idx] != -1) {
            int polar_idx = cart_to_polar_[cart_idx];
            if (bins_mask_[polar_idx]) {
                *(ptr + cart_idx) = 255;
            }
        }
    }

}

void SonarHolder::LinearPolarToCartesianImage(const std::vector<float>& bins, cv::OutputArray _dst) const {
    _dst.create(cart_size_, CV_32FC1);
    cv::Mat dst = _dst.getMat();
    dst.setTo(0);
    float *dst_ptr = reinterpret_cast<float*>(dst.data);
    for (size_t cart_idx = 0; cart_idx < cart_to_polar_.size(); cart_idx++) {
        if (cart_to_polar_[cart_idx] != -1) {
            int polar_idx = cart_to_polar_[cart_idx];
            if (bins_mask_[polar_idx]) {
                *(dst_ptr + cart_idx) = bins[polar_idx];
            }
        }
    }
}

void SonarHolder::WeightedPolarToCartesianImage(const std::vector<float>& bins, cv::OutputArray _dst) const {
    _dst.create(cart_size_, CV_32FC1);
    cv::Mat dst = _dst.getMat();
    dst.setTo(0);

    float *dst_ptr = reinterpret_cast<float*>(dst.data);

    for (size_t cart_idx = 0; cart_idx < cart_to_polar_.size(); cart_idx++) {

        if (cart_to_polar_[cart_idx] != -1) {

            int polar_idx = cart_to_polar_[cart_idx];

            if (bins_mask_[polar_idx]) {

                int beam = polar_idx / bin_count_;
                int bin = polar_idx % bin_count_;

                if (beam < beam_count_-1 && bin < bin_count_-1) {
                    float s0 = bins[(beam+0)*bin_count_+bin+0];
                    float s1 = bins[(beam+0)*bin_count_+bin+1];
                    float s2 = bins[(beam+1)*bin_count_+bin+0];
                    float s3 = bins[(beam+1)*bin_count_+bin+1];

                    float r0 = (bin+0) * cart_height_factor_;
                    float r1 = (bin+1) * cart_height_factor_;
                    float t0 = bearings_[beam+0];
                    float t1 = bearings_[beam+1];

                    float r = radius_[cart_idx];
                    float t = angles_[cart_idx];

                    float v0 = s0 + (s1 - s0) * (r - r0);
                    float v1 = s2 + (s3 - s2) * (r - r0);
                    float v = v0 + (v1 - v0) * (t - t0) / (t1 - t0);

                    *(dst_ptr + cart_idx) = v;
                }
            }
        }
    }
}

void SonarHolder::SetCartesianToPolarSector(uint32_t polar_idx) {
    uint32_t beam = polar_idx / bin_count_;
    uint32_t bin = polar_idx % bin_count_;
    if (beam >= beam_count_ - 1 || bin >= bin_count_ - 1) return;

    cv::Mat_<cv::Point2f> points(1, 4, cv::Point2f(0, 0));
    points(0) = cart_points_[(beam + 0) * bin_count_ + (bin + 0)];
    points(1) = cart_points_[(beam + 1) * bin_count_ + (bin + 1)];
    points(2) = cart_points_[(beam + 0) * bin_count_ + (bin + 1)];
    points(3) = cart_points_[(beam + 1) * bin_count_ + (bin + 0)];

    cv::Rect rc = cv::boundingRect(points);

    float r0 = bin * cart_height_factor_;
    float r1 = (bin+1) * cart_height_factor_;
    float t0 = bearings_[beam];
    float t1 = bearings_[beam+1];

    cart_center_points_[polar_idx] = base::MathUtil::to_cartesianf(t0 + (t1 - t0) / 2, r0 + (r1 - r0) / 2, -M_PI_2) + cart_origin_;

    for (uint32_t y = rc.tl().y; y <= rc.br().y && y < cart_size_.height; y++) {
        for (uint32_t x = rc.tl().x; x <= rc.br().x && x < cart_size_.width; x++) {
            size_t cart_idx = y * cart_size_.width + x;

            if (cart_to_polar_[cart_idx] == -1) {
                float dx = cart_origin_.x - x;
                float dy = cart_origin_.y - y;
                float r = sqrt(dx * dx + dy * dy);
                float t = atan2(dy, dx) - M_PI_2;

                radius_[cart_idx] = r;
                angles_[cart_idx] = t;

                if (r <= r1 && r >= r0 && t >= t0 && t <= t1) {
                    cart_to_polar_[cart_idx] = polar_idx;
                }
            }
        }
    }
}

void SonarHolder::GetNeighborhood(int polar_index, std::vector<int>& neighbors_indices, int neighbor_size) const {
    size_t total_neighbors = neighbor_size * neighbor_size;

    uint32_t beam = polar_index / bin_count_;
    uint32_t bin = polar_index % bin_count_;

    cv::Point2f point = cv::Point(-1, -1);

    if (neighbors_indices.empty()) {
        neighbors_indices.resize(total_neighbors, -1);
    }

    int j = 0;
    int neighbor_size_2 = neighbor_size / 2;

    for (int i = 0; i < total_neighbors; i++) {
        int x = (i % neighbor_size) - neighbor_size_2;
        int y = (i / neighbor_size) - neighbor_size_2;
        int bi = (beam+y < 0 || beam+y >= beam_count_) ? -1 : beam+y;
        int bj = (bin+x < 0 || bin+x >= bin_count_) ? -1 : bin+x;
        int idx = bi * bin_count_ + bj;

        if (bi != -1 && bj != -1 && idx != polar_index) {
            neighbors_indices[i] = idx;
        }
    }
}

std::vector<cv::Point2f> SonarHolder::GetSectorPoints(int polar_index) const {
    int beam = index_to_beam(polar_index);
    int bin = index_to_bin(polar_index);

    std::vector<cv::Point2f> points(4);

    points[0] = cart_points_[(beam + 0) * bin_count_ + (bin + 0)];
    points[1] = cart_points_[(beam + 1) * bin_count_ + (bin + 1)];
    points[2] = cart_points_[(beam + 0) * bin_count_ + (bin + 1)];
    points[3] = cart_points_[(beam + 1) * bin_count_ + (bin + 0)];

    return points;
}

void SonarHolder::GetPolarLimits(int polar_index, float& start_bin, float& final_bin, float& start_beam, float& final_beam) const {
    int bin = index_to_bin(polar_index);
    int beam = index_to_beam(polar_index);
    start_bin = float(bin+0);
    final_bin = float(bin+1);
    start_beam = bearings_[beam+0];
    final_beam = bearings_[beam+1];
}

void SonarHolder::CopyTo(SonarHolder& out) const {
    CopyHeaderData(out);
    CopyBinsValues(out);
    raw_image_.copyTo(out.raw_image_);
    cart_image_.copyTo(out.cart_image_);
    cart_image_mask_.copyTo(out.cart_image_mask_);
}

void SonarHolder::CopyHeaderData(SonarHolder& out) const {
    out.bearings_ = bearings_;
    out.bin_count_ = bin_count_;
    out.beam_count_ = beam_count_;
    out.beam_width_ = beam_width_;
    out.total_elements_ = total_elements_;

    out.cart_size_ = cart_size_;
    out.cart_origin_ = cart_origin_;
    out.cart_points_ = cart_points_;
    out.cart_center_points_ = cart_center_points_;
    out.cart_to_polar_ = cart_to_polar_;

    out.radius_ = radius_;
    out.angles_ = angles_;
}

void SonarHolder::CopyBinsValues(SonarHolder& out) const {
    out.bins_ = bins_;
    out.bins_mask_ = bins_mask_;
}

} /* namespace sonar_processing */
