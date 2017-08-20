#ifndef sonar_processing_HOGDetector_hpp
#define sonar_processing_HOGDetector_hpp

#include <vector>
#include <base/samples/Sonar.hpp>
#include <opencv2/opencv.hpp>
#include "SonarHolder.hpp"
#include "SonarImagePreprocessing.hpp"

namespace sonar_processing {

class HOGDetector
{

public:

    HOGDetector();
    ~HOGDetector();

    void Train(
        const std::vector<base::samples::Sonar>& training_samples,
        const std::vector<std::vector<cv::Point> >& training_annotations,
        const std::string& training_filename);

    void set_windown_size(cv::Size size) {
        window_size_ = size;
    }

    void set_show_descriptor(bool show_descriptor) {
        show_descriptor_ = show_descriptor;
    }

    void set_training_scale_factor(double training_scale_factor) {
        training_scale_factor_ = training_scale_factor;
    }

    void LoadSVMTrain(const std::string& svm_model_filename);

    bool Detect(
        const base::samples::Sonar& sample,
        const std::vector<cv::Point>& annotation_points,
        std::vector<cv::RotatedRect>& locations,
        std::vector<double>& found_weights);

    bool Detect(
        const cv::Mat& sonar_source_image,
        const cv::Mat& sonar_source_mask,
        const std::vector<cv::Point>& annotation_points,
        std::vector<cv::RotatedRect>& locations,
        std::vector<double>& found_weights);

private:

    void LoadTrainingData(
        const std::vector<base::samples::Sonar>& training_samples,
        const std::vector<std::vector<cv::Point> >& training_annotations,
        std::vector<cv::Mat>& gradient_positive,
        std::vector<cv::Mat>& gradient_negative);

    void PrepareInput(
        const std::vector<cv::Point>& annotation,
        cv::Mat& input_image,
        cv::Mat& input_mask,
        cv::Mat& annotation_mask,
        double& rotated_angle);

    void ComputeTrainingData(
        const std::vector<cv::Point>& annotation,
        std::vector<cv::Mat>& gradient_positive,
        std::vector<cv::Mat>& gradient_negative);

    void CreateAnnotationMask(
        const cv::Size& size,
        const std::vector<cv::Point>& annotation,
        cv::Mat& annotation_mask);


    void OrientationNormalize(
        const cv::Mat& source_image,
        const cv::Mat& source_mask,
        const cv::Mat& annotation_mask,
        cv::RotatedRect bbox,
        cv::Mat& rotated_image,
        cv::Mat& rotated_mask,
        cv::Mat& rotated_annotation_mask,
        double& rotated_angle);

    void PreparePositiveInput(
        const cv::Mat& source_image,
        const cv::Mat& annotation_mask,
        cv::Mat& result_image);

    void ComputePositive(
        const cv::Mat& source_image,
        const cv::Mat& annotation_mask,
        std::vector<cv::Mat>& gradient_list_positive);

    void ComputeNegative(
        const cv::Mat& source_image,
        const cv::Mat& source_mask,
        const cv::Mat& annotation_mask,
        std::vector<cv::Mat>& gradient_list_negative);

    void PrepareNegativeInput(
        const cv::Mat& source_image,
        const cv::Mat& source_mask,
        const cv::Mat& annotation_mask,
        cv::Mat& result_image,
        cv::Mat& result_mask,
        cv::Mat& result_annotation_mask);


    void ComputeNegativeGradient(
        const cv::Mat& src,
        const cv::Mat& mask,
        const cv::Mat& annotation_mask,
        std::vector<cv::Mat>& gradient_list_negative);

    void ComputeGradient(
        const cv::Mat& source_image,
        std::vector<cv::Mat>& gradient_list);

    void PrepareTrainingData(
        const std::vector<cv::Mat>& positive,
        const std::vector<cv::Mat>& negative,
        std::vector<int>& labels,
        cv::Mat& training_data);

    void SVMTrain(
        const std::vector<int>& labels,
        const cv::Mat& training_data,
        const std::string& training_filename);

    void TransformLocation(
        const std::vector<cv::Rect>& locations,
        double scale,
        double rotate,
        cv::Point translate,
        cv::Size source_size,
        std::vector<cv::RotatedRect>& rotated_locations);

    SonarHolder sonar_holder_;
    SonarImagePreprocessing sonar_image_processing_;
    cv::Size window_size_;

    double training_scale_factor_;
    bool show_descriptor_;

    cv::HOGDescriptor hog_descriptor_;
    cv::Mat sonar_source_image_;
    cv::Mat sonar_source_mask_;

};

} /* namespace sonar_processing*/


#endif /* SonarPreprocessing_hpp */
