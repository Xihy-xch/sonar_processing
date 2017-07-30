#include "Filtering.hpp"

namespace sonar_processing {

namespace filtering {

FilterApplier::FilterApplier(const SonarHolder& sonar_holder)
    : sonar_holder_(sonar_holder)
{
    if (!sonar_holder_.has_neighborhood_table()) {
        throw std::invalid_argument("The Neighborhood Table was not built.");
    }
}

void FilterApplier::PerformConvolution(std::vector<float>& dst, const std::vector<float>& kernel, int ksize) const {
    if (dst.empty()) {
        dst.assign(sonar_holder_.total_elements(), 0);
    }

    std::vector<int> indices(ksize * ksize, -1);
    std::vector<float> values(ksize * ksize, 0);

    for (int index= 0; index < sonar_holder_.total_elements(); index++) {
        if (validate_index(index)) {
            sonar_holder_.GetCartesianNeighborhoodIndices(index, indices, ksize);
            if (validate_neighborhood_indices(indices)) {
                sonar_holder_.values(indices, values);
                dst[index] = apply_kernel(values, kernel);
            }
        }
    }
}

} /* namespace filtering */

} /* namespace sonar_processing */
