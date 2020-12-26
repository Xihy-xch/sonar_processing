#ifndef LinearSVM_hpp
#define LinearSVM_hpp

#include <opencv2/opencv.hpp>

class LinearSVM : public cv::SVM {

public:
    void get_detector(std::vector<float> &detector) {
        const CvSVMDecisionFunc *df = decision_func;
        const double *alphas = df->alpha;
        double rho = df->rho;

        int sv_count = get_support_vector_count();
        int var_count = get_var_count();

        detector.clear();
        detector.resize(var_count, 0);

        for (int i = 0; i < sv_count; i++) {
            const float *v = get_support_vector(i);
            float alpha = alphas[i];
            for (int j = 0; j < var_count; j++, v++) {
                detector[j] += (alpha) * (*v);
            }
        }
        detector.push_back((float) -rho);
    }
};

#endif
