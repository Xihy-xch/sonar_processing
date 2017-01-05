#ifndef sonar_processing_Filtering_hpp
#define sonar_processing_Filtering_hpp

#include <vector>
#include "sonar_processing/BasicOperations.hpp"
#include "sonar_processing/ImageUtil.hpp"
#include "sonar_processing/SonarHolder.hpp"

namespace sonar_processing {

namespace filtering {

class FilterApplier {

public:

    FilterApplier(const SonarHolder& sonar_holder);

    virtual ~FilterApplier() {
    }

    void PerformConvolution(std::vector<float>& dst, const std::vector<float>& kernel, int ksize = 3) const;

private:

    bool validate_index(int index) const {
        int bin = sonar_holder_.index_to_bin(index);
        int beam = sonar_holder_.index_to_beam(index);
        const std::vector<uchar>& mask = sonar_holder_.bins_mask();

        return bin >= 1 && bin <= sonar_holder_.bin_count()-2 &&
               beam >= 1 && beam <= sonar_holder_.beam_count()-2 &&
               mask[index] != 0;
    }

    bool validate_neighborhood_indices(const std::vector<int>& indices) const {
        return (std::find(indices.begin(), indices.end(), -1) == indices.end());
    }

    float apply_kernel(const std::vector<float>& src, const std::vector<float>& kernel) const {
        float result = 0;
        for (size_t i = 0; i < kernel.size(); i++) result += kernel[i] * src[i];
        return result;
    }

    const SonarHolder& sonar_holder_;

};

inline void filter2d(const SonarHolder& sonar_holder, std::vector<float>& dst, const std::vector<float>& kernel, int ksize = 3) {
    FilterApplier filter_applier(sonar_holder);
    filter_applier.PerformConvolution(dst, kernel, ksize);
}

inline void filter2d(const SonarHolder& sonar_holder, std::vector<float>& dst, const cv::Mat& kernel) {
    assert(kernel.depth() == CV_32F);
    filter2d(sonar_holder, dst, image_util::mat2vector<float>(kernel), kernel.rows);

}
} /* namespace filtering */

} /* namespace sonar_processing */

#endif /* SonarHolder_hpp */
