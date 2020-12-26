#ifndef sonar_processing_HOGDetector_hpp
#define sonar_processing_HOGDetector_hpp

#include <vector>
#include <base/samples/Sonar.hpp>
#include <opencv2/opencv.hpp>
#include "SonarHolder.hpp"
#include "SonarImagePreprocessing.hpp"

namespace sonar_processing {

    class HOGDetector {

    public:

        HOGDetector();

        ~HOGDetector();

        void Train(
                const std::vector<base::samples::Sonar> &training_samples,
                const std::vector<std::vector<cv::Point> > &training_annotations,
                const std::string &training_filename);

        void set_windown_size(cv::Size size) {
            window_size_ = size;
        }

        void reset_detection_stats() {
            succeeded_detect_count_ = 0;
            failed_detect_count_ = 0;
            memset(&last_detected_location_, 0, sizeof(last_detected_location_));
        }

        void set_show_descriptor(bool show_descriptor) {
            show_descriptor_ = show_descriptor;
        }

        void set_training_scale_factor(double training_scale_factor) {
            training_scale_factor_ = training_scale_factor;
        }

        void set_show_positive_window(bool show_positive_window) {
            show_positive_window_ = show_positive_window;
        }

        void set_sonar_image_size(cv::Size sonar_image_size) {
            sonar_image_size_ = sonar_image_size;
        }

        void set_image_scale(double image_scale) {
            image_scale_ = image_scale;
        }

        void set_window_stride(const cv::Size &window_stride) {
            window_stride_ = window_stride;
        }

        void set_positive_input_validate(bool positive_input_validate) {
            positive_input_validate_ = positive_input_validate;
        }

        void set_sonar_image_processing(const SonarImagePreprocessing &p) {
            sonar_image_processing_.set_roi_extract_thresh(p.roi_extract_thresh());
            sonar_image_processing_.set_mean_difference_filter_enable(p.mean_difference_filter_enable());
            sonar_image_processing_.set_roi_extract_start_bin(p.roi_extract_start_bin());
            sonar_image_processing_.set_mean_filter_ksize(p.mean_filter_ksize());
            sonar_image_processing_.set_mean_difference_filter_ksize(p.mean_difference_filter_ksize());
            sonar_image_processing_.set_mean_difference_filter_source(p.mean_difference_filter_source());
            sonar_image_processing_.set_median_blur_filter_ksize(p.median_blur_filter_ksize());
            sonar_image_processing_.set_border_filter_type(p.border_filter_type());
            sonar_image_processing_.set_border_filter_enable(p.border_filter_enable());
        }

        void set_detection_scale_factor(double detection_scale_factor) {
            detection_scale_factor_ = detection_scale_factor;
        }

        void set_detection_minimum_weight(double detection_minimum_weight) {
            detection_minimum_weight_ = detection_minimum_weight;
        }

        void set_orientation_step(double orientation_step) {
            orientation_step_ = orientation_step;
        }

        void set_orientation_range(double orientation_range) {
            orientation_range_ = orientation_range;
        }

        void LoadSVMTrain(const std::string &svm_model_filename);

        bool Detect(
                const base::samples::Sonar &sample,
                const std::vector<cv::Point> &annotation_points,
                std::vector<cv::RotatedRect> &locations,
                std::vector<double> &found_weights);

        bool Detect(
                const cv::Mat &sonar_source_image,
                const cv::Mat &sonar_source_mask,
                const std::vector<cv::Point> &annotation_points,
                std::vector<cv::RotatedRect> &locations,
                std::vector<double> &found_weights);

        bool Detect(
                const cv::Mat &sonar_source_image,
                const cv::Mat &sonar_source_mask,
                std::vector<cv::RotatedRect> &locations,
                std::vector<double> &found_weights);


    private:

        void LoadTrainingData(
                const std::vector<base::samples::Sonar> &training_samples,
                const std::vector<std::vector<cv::Point> > &training_annotations,
                std::vector<cv::Mat> &gradient_positive,
                std::vector<cv::Mat> &gradient_negative);

        void PerformPreprocessing(
                cv::Mat &preprocessed_image,
                cv::Mat &preprocessed_mask);

        void PrepareInput(
                const cv::Mat &preprocessed_image,
                const cv::Mat &preprocessed_mask,
                const std::vector<cv::Point> &annotation,
                double scale_factor,
                cv::Mat &input_image,
                cv::Mat &input_mask,
                cv::Mat &annotation_mask,
                double &rotated_angle);

        void PrepareInput(
                const cv::Mat &preprocessed_image,
                const cv::Mat &preprocessed_mask,
                double scale_factor,
                cv::Mat &input_image,
                cv::Mat &input_mask);


        void ComputeTrainingData(
                const std::vector<cv::Point> &annotation,
                std::vector<cv::Mat> &gradient_positive,
                std::vector<cv::Mat> &gradient_negative);

        void CreateAnnotationMask(
                const cv::Size &size,
                const std::vector<cv::Point> &annotation,
                cv::Mat &annotation_mask);


        void OrientationNormalize(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                const cv::Mat &annotation_mask,
                cv::RotatedRect bbox,
                cv::Mat &rotated_image,
                cv::Mat &rotated_mask,
                cv::Mat &rotated_annotation_mask,
                double &rotated_angle);

        void PreparePositiveInput(
                const cv::Mat &source_image,
                const cv::Mat &annotation_mask,
                cv::Mat &result_image);

        void ComputePositive(
                const cv::Mat &source_image,
                const cv::Mat &annotation_mask,
                std::vector<cv::Mat> &gradient_list_positive);

        void ComputeNegative(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                const cv::Mat &annotation_mask,
                std::vector<cv::Mat> &gradient_list_negative);

        void PrepareNegativeInput(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                const cv::Mat &annotation_mask,
                cv::Mat &result_image,
                cv::Mat &result_mask,
                cv::Mat &result_annotation_mask);


        void ComputeNegativeGradient(
                const cv::Mat &src,
                const cv::Mat &mask,
                const cv::Mat &annotation_mask,
                std::vector<cv::Mat> &gradient_list_negative);

        void ComputeGradient(
                const cv::Mat &source_image,
                std::vector<cv::Mat> &gradient_list);

        void PrepareTrainingData(
                const std::vector<cv::Mat> &positive,
                const std::vector<cv::Mat> &negative,
                std::vector<int> &labels,
                cv::Mat &training_data);

        void SVMTrain(
                const std::vector<int> &labels,
                const cv::Mat &training_data,
                const std::string &training_filename);

        void TransformLocation(
                const std::vector<cv::Rect> &locations,
                double scale,
                double rotate,
                cv::Point translate,
                cv::Size source_size,
                std::vector<cv::RotatedRect> &rotated_locations);

        void ResizeAnnotationPoints(
                const std::vector<cv::Point> &source_points,
                std::vector<cv::Point> &result_points);


        void FilterLocationInsideMask(
                const std::vector<cv::Rect> &locations,
                const std::vector<double> &weights,
                std::vector<cv::Rect> &result_locations,
                std::vector<double> &result_weights,
                const cv::Mat &input,
                const cv::Mat &mask);

        bool ValidatePositiveInput(
                const cv::Mat &mask,
                const cv::Mat &annotation_mask);

        void RotateInput(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                const cv::Point2f &center,
                double angle,
                cv::Mat &rotated_image,
                cv::Mat &rotated_mask);

        bool PerformDetect(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                double rotated_angle,
                std::vector<cv::RotatedRect> &locations,
                std::vector<double> &found_weights);

        void RotateAndDetect(
                const cv::Mat &source_image,
                const cv::Mat &source_mask,
                double first_angle,
                double last_angle,
                double angle_step,
                std::vector<cv::RotatedRect> &locations,
                std::vector<double> &found_weights);


        void FindBestDetectionLocation(
                const std::vector<cv::RotatedRect> &locations,
                const std::vector<double> &weights,
                double &best_weight,
                cv::RotatedRect &best_location);

        cv::Rect GetLastDetectedBoundingRect(
                double scale,
                cv::Size max_size);

        SonarHolder sonar_holder_;
        SonarImagePreprocessing sonar_image_processing_;
        cv::Size window_size_;

        double training_scale_factor_;
        double detection_scale_factor_;
        double detection_minimum_weight_;

        bool show_descriptor_;
        bool show_positive_window_;
        bool positive_input_validate_;

        cv::HOGDescriptor hog_descriptor_;
        cv::Mat sonar_source_image_;
        cv::Mat sonar_source_mask_;

        cv::Size sonar_image_size_;

        double image_scale_;
        cv::Size window_stride_;

        double orientation_step_;
        double orientation_range_;

        cv::RotatedRect last_detected_location_;

        int succeeded_detect_count_;
        int failed_detect_count_;
    };

} /* namespace sonar_processing*/


#endif /* SonarPreprocessing_hpp */
