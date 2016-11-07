#ifndef sonar_processing_SonarHolder_hpp
#define sonar_processing_SonarHolder_hpp

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "sonar_processing/ImageUtils.hpp"

namespace sonar_processing {

class SonarHolder {

public:

    enum PolarToCartesianInterpolationType {
        LINEAR = 0,
        WEIGHTED = 1
    };

    SonarHolder();

    SonarHolder(std::vector<float> bins,
                float start_beam,
                float beam_width,
                uint32_t bin_count,
                uint32_t beam_count,
                int interpolation_type = WEIGHTED);

    SonarHolder(std::vector<float> bins,
                std::vector<float> bearings,
                float beam_width,
                uint32_t bin_count,
                uint32_t beam_count,
                int interpolation_type = WEIGHTED);

    ~SonarHolder();

    void Reset(std::vector<float> bins,
               std::vector<float> bearings,
               float beam_width,
               uint32_t bin_count,
               uint32_t beam_count);

    void Reset(std::vector<float> bins,
               float start_beam,
               float beam_width,
               uint32_t bin_count,
               uint32_t beam_count);
               
    void ResetBins(std::vector<float> bins);


    std::vector<float> bins() const {
        return bins_;
    }
    
    void set_bins(std::vector<float> bins) {
        bins_ = bins;
        std::copy(bins.begin(), bins.end(), bins_.begin());
    }

    std::vector<uchar> bins_mask() const {
        return bins_mask_;
    }

    float value_at(int index) const {
        return bins_[index];
    }

    float value_at(uint32_t bin, uint32_t beam) const {
        return bins_[beam * bin_count_ + bin];
    }

    void values(const std::vector<int>& indices, std::vector<float>& values) const {
        values.assign(indices.size(), 0.0);
        for (size_t i = 0; i < indices.size(); i++) {
            if (indices[i] > 0) values[i] = bins_[indices[i]];
        }
    }

    float beam_value_at(int beam) {
        return bearings_[beam];
    }

    const std::vector<float>& bearings() const {
        return bearings_;
    }

    uint32_t bin_count() const {
        return bin_count_;
    }

    uint32_t beam_count() const  {
        return beam_count_;
    }
    
    uint32_t total_elements() const {
        return total_elements_;
    }

    float beam_width() const {
        return beam_width_;
    }

    float beam_step() const {
        return beam_width_ / (float)beam_count_;
    }

    std::vector<cv::Point2f> cart_points() const {
        return cart_points_;
    }

     void cart_points(const std::vector<int>& indices, std::vector<cv::Point2f>& points) const {
         points.resize(indices.size());
        for (int i = 0; i < indices.size(); i++) points[i] = cart_points_[indices[i]];
    }

    cv::Point2f cart_center_point(int index) const  {
        return cart_center_points_[index];
    }

    std::vector<cv::Point2f> cart_center_points() const {
        return cart_center_points_;
    }

    cv::Size cart_size() const {
        return cart_size_;
    }

    cv::Point2f cart_origin() const {
        return cart_origin_;
    }

    cv::Mat cart_image() const {
        return cart_image_;
    }
    
    cv::Mat raw_image() const {
        return raw_image_;
    }

    cv::Point2f cart_point(uint32_t bin, uint32_t beam) const {
        return cart_points_[beam * bin_count_ + bin];
    }

    cv::Point2f cart_center_point(uint32_t bin, uint32_t beam) const {
        return cart_center_points_[beam * bin_count_ + bin];
    }

    int index_to_beam(int index) const {
        return index / bin_count_;
    }

    int index_to_bin(int index) const {
        return index % bin_count_;
    }

    void index_to_polar(int index, int& bin, int& beam) const {
        beam = index_to_beam(index);
        bin = index_to_bin(index);
    }

    int index_at(int beam, int bin) const {
        return beam * bin_count_ + bin;
    }

    cv::Rect cart_bounding_rect(uint32_t bin0, uint32_t beam0, uint32_t bin1, uint32_t beam1) const {
        std::vector<cv::Point2f> pts(4);
        pts[0] = cart_point(bin0, beam0);
        pts[1] = cart_point(bin1, beam0);
        pts[2] = cart_point(bin0, beam1);
        pts[3] = cart_point(bin1, beam1);
        return cv::boundingRect(cv::Mat(pts));
    }

    void GetNeighborhood(int polar_index, std::vector<int>& neighbors_indices, int neighbor_size = 3) const;

    cv::Point2f sector_top_left_point(int polar_index) const {
        return cart_points_[(index_to_beam(polar_index) + 0) * bin_count_ + (index_to_bin(polar_index) + 0)];
    }

    cv::Point2f sector_top_right_point(int polar_index) const {
        return cart_points_[(index_to_beam(polar_index) + 1) * bin_count_ + (index_to_bin(polar_index) + 0)];
    }

    cv::Point2f sector_bottom_left_point(int polar_index) const {
        return cart_points_[(index_to_beam(polar_index) + 0) * bin_count_ + (index_to_bin(polar_index) + 1)];
    }

    cv::Point2f sector_bottom_right_point(int polar_index) const {
        return cart_points_[(index_to_beam(polar_index) + 1) * bin_count_ + (index_to_bin(polar_index) + 1)];
    }

    std::vector<cv::Point2f> GetSectorPoints(int polar_index) const;

    void GetPolarLimits(int polar_index, float& start_bin, float& final_bin, float& start_beam, float& final_beam) const;

    cv::Rect_<float> sector_bounding_rect(int polar_index) const {
        return image_utils::bounding_rect(GetSectorPoints(polar_index));
    }

    void CopyTo(SonarHolder& out) const;
    
    void CopyTo(SonarHolder& out, const std::vector<int>& start_line_indices, const std::vector<int>& final_line_indices) const;

    void CopyHeaderData(SonarHolder& out) const;

    void CopyBinsValues(SonarHolder& out) const;

    void CopyBinsValues(SonarHolder& out, const std::vector<int>& start_line_indices, const std::vector<int>& final_line_indices) const;
    
    void SetBinsOfInterest(const std::vector<int>& start_line_indices, const std::vector<int>& final_line_indices);

private:

    void Initialize();
    void InitializeCartesianPoints();
    void InitializePolarMapping();
    void InitializeCartesianImage();
    void SetCartesianToPolarSector(uint32_t polar_idx);
    void LinearPolarToCartesianImage(cv::OutputArray dst);
    void WeightedPolarToCartesianImage(cv::OutputArray dst);    

    std::vector<float> BuildBeamBearings(float start_beam, float beam_width, uint32_t beam_count);

    std::vector<float> bins_;
    std::vector<float> bearings_;
    
    std::vector<uchar> bins_mask_;

    uint32_t bin_count_;
    uint32_t beam_count_;
    uint32_t total_elements_;
    float beam_width_;

    std::vector<cv::Point2f> cart_points_;
    std::vector<cv::Point2f> cart_center_points_;

    std::vector<int> cart_to_polar_;
    std::vector<float> radius_;
    std::vector<float> angles_;

    cv::Size cart_size_;
    cv::Point2f cart_origin_;

    int interpolation_type_;

    cv::Mat cart_image_;
    cv::Mat raw_image_;
};

} /* namespace sonar_processing */


#endif /* sonar_util_SonarHolder_hpp */
