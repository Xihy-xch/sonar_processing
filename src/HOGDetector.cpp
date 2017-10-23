#include <cstdio>
#include "HogDescriptorViz.hpp"
#include "LinearSVM.hpp"
#include "Preprocessing.hpp"
#include "Utils.hpp"
#include "HOGDetector.hpp"

namespace sonar_processing {

HOGDetector::HOGDetector()
{
    window_size_ = cv::Size(192, 48);
    window_stride_ = cv::Size(8, 8);
    training_scale_factor_ = 0.3;
    detection_scale_factor_ = 0.5;
    detection_minimum_weight_ = 0;
    show_descriptor_ = false;
    show_positive_window_ = false;
    sonar_image_size_ = cv::Size(-1, -1);
    image_scale_ = 1.125;
    orientation_step_ = 15.0;
    orientation_range_ = 15.0;
    succeeded_detect_count_ = 0;
    failed_detect_count_ = 0;
    memset(&last_detected_location_, 0, sizeof(last_detected_location_));
}

HOGDetector::~HOGDetector() {

}

void HOGDetector::Train(
    const std::vector<base::samples::Sonar>& training_samples,
    const std::vector<std::vector<cv::Point> >& training_annotations,
    const std::string& training_filename)
{
    // set hog window size
    hog_descriptor_.winSize = window_size_;

    //load training data
    std::vector<cv::Mat> gradient_positive;
    std::vector<cv::Mat> gradient_negative;
    LoadTrainingData(training_samples, training_annotations, gradient_positive, gradient_negative);

    // prepare training data
    std::vector<int> labels;
    cv::Mat training_data;
    PrepareTrainingData(gradient_positive, gradient_negative, labels, training_data);

    if (show_positive_window_) {
        cv::destroyWindow("positive_input_image");
    }

    // training using the hog descriptor
    SVMTrain(labels, training_data, training_filename);
}

void HOGDetector::LoadSVMTrain(const std::string& svm_model_filename)
{
    LinearSVM svm;
    svm.load(svm_model_filename.c_str());

    std::vector<float> hog_detector;
    svm.get_detector(hog_detector);

    hog_descriptor_.winSize = window_size_;
    hog_descriptor_.setSVMDetector(hog_detector);
}

bool HOGDetector::Detect(
    const base::samples::Sonar& sample,
    const std::vector<cv::Point>& annotation_points,
    std::vector<cv::RotatedRect>& locations,
    std::vector<double>& found_weights)
{
    if (annotation_points.empty()) {
        return false;
    }

    sonar_holder_.Reset(
        sample.bins,
        utils::get_radians(sample.bearings),
        sample.beam_width.getRad(),
        sample.bin_count,
        sample.beam_count,
        sonar_image_size_);

    return Detect(
        sonar_holder_.cart_image(),
        sonar_holder_.cart_image_mask(),
        annotation_points,
        locations,
        found_weights);
}

bool HOGDetector::Detect(
    const cv::Mat& sonar_source_image,
    const cv::Mat& sonar_source_mask,
    const std::vector<cv::Point>& annotation_points,
    std::vector<cv::RotatedRect>& locations,
    std::vector<double>& found_weights)
{
    sonar_source_image.copyTo(sonar_source_image_);
    sonar_source_mask.copyTo(sonar_source_mask_);

    // perform preprocessing
    cv::Mat preprocessed_image;
    cv::Mat preprocessed_mask;
    PerformPreprocessing(preprocessed_image, preprocessed_mask);

    double rotated_angle;
    cv::Mat input_image;
    cv::Mat input_mask;
    cv::Mat input_annotation_mask;

    // prepare hog inputs
    PrepareInput(
        preprocessed_image,
        preprocessed_mask,
        annotation_points,
        detection_scale_factor_,
        input_image,
        input_mask,
        input_annotation_mask,
        rotated_angle);

    return PerformDetect(input_image, input_mask, rotated_angle, locations, found_weights);
}

bool HOGDetector::Detect(
    const cv::Mat& sonar_source_image,
    const cv::Mat& sonar_source_mask,
    std::vector<cv::RotatedRect>& locations,
    std::vector<double>& found_weights)
{

    const int SUCCEDED_LIMIT = 3;
    const int FAILED_LIMIT = 10;
    const float MIN_LOCATION_DISTANCE = 50.0f;
    const float PARTIAL_ROTATE_STEP = 5;
    const float COMPLETE_START_RANGE_ANGLE = -90;
    const float COMPLETE_FINAL_RANGE_ANGLE = 90;

    sonar_source_image.copyTo(sonar_source_image_);
    sonar_source_mask.copyTo(sonar_source_mask_);

    // perform preprocessing
    cv::Mat preprocessed_image;
    cv::Mat preprocessed_mask;
    PerformPreprocessing(preprocessed_image, preprocessed_mask);

    cv::Mat scaled_image;
    cv::resize(preprocessed_image, scaled_image, cv::Size(), detection_scale_factor_, detection_scale_factor_);

    cv::Mat scaled_mask;
    cv::resize(preprocessed_mask, scaled_mask, cv::Size(), detection_scale_factor_, detection_scale_factor_);

    if (failed_detect_count_ > FAILED_LIMIT) {
        succeeded_detect_count_ = 0;
        failed_detect_count_ = 0;
    }

    if (succeeded_detect_count_ < SUCCEDED_LIMIT) {

        RotateAndDetect(
            scaled_image,
            scaled_mask,
            COMPLETE_START_RANGE_ANGLE,
            COMPLETE_FINAL_RANGE_ANGLE,
            orientation_step_,
            locations,
            found_weights);

        if (locations.empty()) {
            succeeded_detect_count_ = 0;
            return false;
        }

        double best_weight = -1;
        cv::RotatedRect best_detected_location;
        FindBestDetectionLocation(locations, found_weights, best_weight, best_detected_location);

        if (best_weight < detection_minimum_weight_) {
            found_weights.clear();
            locations.clear();
            succeeded_detect_count_ = 0;
            return false;
        }

        if (succeeded_detect_count_ >= 1) {
            cv::Point2f point_dt = best_detected_location.center-last_detected_location_.center;
            float dt = sqrt(point_dt.x * point_dt.x + point_dt.y * point_dt.y);

            if (dt > MIN_LOCATION_DISTANCE) {
                found_weights.clear();
                locations.clear();
                succeeded_detect_count_ = 0;
                return false;
            }
        }

        last_detected_location_ = best_detected_location;

        succeeded_detect_count_++;
        failed_detect_count_ = 0;

        found_weights.clear();
        locations.clear();

        return false;
    }

    cv::Rect bbox = GetLastDetectedBoundingRect(
        detection_scale_factor_,
        scaled_mask.size());

    cv::Mat new_scaled_mask = cv::Mat::zeros(scaled_mask.size(), scaled_mask.type());
    std::cout << "size: " << scaled_mask.size() << std::endl;
    std::cout << "bbox: " << bbox << std::endl;
    scaled_mask(bbox).copyTo(new_scaled_mask(bbox));

    float start_angle;
    float final_angle;
    float angle_step;

    if (succeeded_detect_count_ % 50 == 0) {
        start_angle = COMPLETE_START_RANGE_ANGLE;
        final_angle = COMPLETE_FINAL_RANGE_ANGLE;
        angle_step = orientation_step_;
    }
    else {
        start_angle = last_detected_location_.angle-orientation_range_;
        final_angle = last_detected_location_.angle+orientation_range_;
        angle_step = PARTIAL_ROTATE_STEP;
    }

    RotateAndDetect(
        scaled_image,
        new_scaled_mask,
        start_angle,
        final_angle,
        angle_step,
        locations,
        found_weights);


    if (locations.empty()) {
        failed_detect_count_++;
        return false;
    }

    double best_weight = -1;

    cv::RotatedRect best_detected_location;
    FindBestDetectionLocation(locations, found_weights, best_weight, best_detected_location);
    //
    // if (best_weight < detection_minimum_weight_) {
    //     found_weights.clear();
    //     locations.clear();
    //     failed_detect_count_++;
    //     return false;
    // }

    cv::Point2f point_dt = best_detected_location.center-last_detected_location_.center;
    float dt = sqrt(point_dt.x * point_dt.x + point_dt.y * point_dt.y);

    if (dt > MIN_LOCATION_DISTANCE) {
        found_weights.clear();
        locations.clear();
        failed_detect_count_++;
        return false;
    }

    last_detected_location_ = best_detected_location;
    succeeded_detect_count_++;
    failed_detect_count_ = 0;

    return true;
}

void HOGDetector::RotateAndDetect(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    double first_angle,
    double last_angle,
    double angle_step,
    std::vector<cv::RotatedRect>& locations,
    std::vector<double>& found_weights)
{
    cv::Mat rotated_image;
    cv::Mat rotated_mask;
    cv::Point2f center = cv::Point2f(source_image.cols/2, source_image.rows/2);

    // cv::imshow("preprocessed_image", source_image);
    // cv::waitKey(15);

    for (double theta=first_angle; theta<=last_angle; theta+=angle_step) {
        if (fabs(theta) >= 2.5) {
            RotateInput(source_image, source_mask, center, theta, rotated_image, rotated_mask);
            PerformDetect(rotated_image, rotated_mask, theta, locations, found_weights);
        }
        else {
            PerformDetect(source_image, source_mask, theta, locations, found_weights);
        }
    }
}

void HOGDetector::FindBestDetectionLocation(
    const std::vector<cv::RotatedRect>& locations,
    const std::vector<double>& weights,
    double& best_weight,
    cv::RotatedRect &best_location)
{
    std::vector<size_t> indices(weights.size());
    for (size_t i = 0; i < indices.size(); i++) indices[i]=i;
    std::sort(indices.begin(), indices.end(), sonar_processing::utils::IndexComparator<double>(weights));
    std::reverse(indices.begin(), indices.end());
    best_weight = weights[indices[0]];
    best_location = locations[indices[0]];
}

bool HOGDetector::PerformDetect(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    double rotated_angle,
    std::vector<cv::RotatedRect>& locations,
    std::vector<double>& found_weights)
{
    cv::Size source_image_size = source_image.size();

    // set region of interest
    cv::Rect bounding_rect = image_util::get_bounding_rect(source_mask);

    // set the region of intereset
    cv::Mat input_image;
    cv::Mat input_mask;
    source_image(bounding_rect).copyTo(input_image);
    source_mask(bounding_rect).copyTo(input_mask);

    // cv::imshow("input_image", input_image);
    // cv::imshow("input_mask", input_mask);
    // cv::waitKey(15);

    if (window_size_.width >= input_image.size().width ||
        window_size_.height >= input_image.size().height) {
        return false;
    }

    // convert to unsigned char
    input_image.convertTo(input_image, CV_8U, 255.0);

    std::vector<cv::Rect> locations_rects;
    std::vector<double> weights;
    hog_descriptor_.detectMultiScale(
        input_image, // the input image
        locations_rects, // the found locations rect
        weights, // the found weights
        0.0, // the hit-threshold
        window_stride_, // the window stride
        cv::Size(8, 8), // the padding
        //
        // 1.125, // the image scale
        image_scale_, // the image scale
        // 2, // the final threshold
        2, // the final threshold
        false); // enable the mean shift grouping

    FilterLocationInsideMask(locations_rects, weights, locations_rects, weights, input_image, input_mask);

    if (locations_rects.empty()){
        return false;
    }

    found_weights.insert(found_weights.end(), weights.begin(), weights.end());

    cv::Point translate = bounding_rect.tl();

    TransformLocation(
        locations_rects, detection_scale_factor_, -rotated_angle,
        translate, source_image_size, locations);

    return true;
}


void HOGDetector::LoadTrainingData(
    const std::vector<base::samples::Sonar>& training_samples,
    const std::vector<std::vector<cv::Point> >& training_annotations,
    std::vector<cv::Mat>& gradient_positive,
    std::vector<cv::Mat>& gradient_negative)
{
    printf("Extracting HOG from samples.\n\033[s");
    for (size_t i = 0; i < training_samples.size(); i++) {
        printf("\033[uSample: %ld of %ld",i, training_samples.size());
        fflush(stdout);

        base::samples::Sonar sample = training_samples[i];

        sonar_holder_.Reset(sample.bins,
            utils::get_radians(sample.bearings),
            sample.beam_width.getRad(),
            sample.bin_count,
            sample.beam_count,
            sonar_image_size_);

        sonar_source_image_ = sonar_holder_.cart_image();
        sonar_source_mask_ = sonar_holder_.cart_image_mask();

        std::vector<cv::Point> annotation_points = training_annotations[i];

        if (sonar_image_size_ != cv::Size(-1, -1)) {
            ResizeAnnotationPoints(annotation_points, annotation_points);
        }

        ComputeTrainingData(annotation_points, gradient_positive, gradient_negative);
    }
    printf("\n");
    std::cout << "Total Positive samples: " << gradient_positive.size() << std::endl;
    std::cout << "Total Negative samples: " << gradient_negative.size() << std::endl;
}

void HOGDetector::PerformPreprocessing(
    cv::Mat& preprocessed_image,
    cv::Mat& preprocessed_mask)
{
    // perform the sonar image preprocessing
    if (sonar_image_size_ != cv::Size(-1, -1)) {
        sonar_image_processing_.Apply(sonar_source_image_, sonar_source_mask_, preprocessed_image, preprocessed_mask);
    }
    else {
        sonar_image_processing_.Apply(sonar_source_image_, sonar_source_mask_, preprocessed_image, preprocessed_mask, 0.5);
    }
}

void HOGDetector::PrepareInput(
    const cv::Mat& preprocessed_image,
    const cv::Mat& preprocessed_mask,
    const std::vector<cv::Point>& annotation,
    double scale_factor,
    cv::Mat& input_image,
    cv::Mat& input_mask,
    cv::Mat& input_annotation_mask,
    double& rotated_angle)
{
    cv::Mat annotation_mask;
    CreateAnnotationMask(sonar_source_image_.size(), annotation, annotation_mask);

    cv::Mat scaled_image;
    cv::resize(preprocessed_image, scaled_image, cv::Size(), scale_factor, scale_factor);

    cv::Mat scaled_mask;
    cv::resize(preprocessed_mask, scaled_mask, cv::Size(), scale_factor, scale_factor);

    if (!annotation.empty()) {
        cv::Mat scaled_annotation_mask;
        cv::resize(annotation_mask, scaled_annotation_mask, cv::Size(), scale_factor, scale_factor);

        // perform the orientation normalize
        OrientationNormalize(scaled_image, scaled_mask, scaled_annotation_mask,
            cv::minAreaRect(annotation),
            input_image, input_mask, input_annotation_mask, rotated_angle);
            return;
    }

    scaled_image.copyTo(input_image);
    scaled_mask.copyTo(input_mask);
    input_annotation_mask = cv::Mat::zeros(scaled_mask.size(), scaled_mask.type());
}


void HOGDetector::ComputeTrainingData(
    const std::vector<cv::Point>& annotation,
    std::vector<cv::Mat>& gradient_positive,
    std::vector<cv::Mat>& gradient_negative)
{
    // perform preprocessing
    cv::Mat preprocessed_image;
    cv::Mat preprocessed_mask;
    PerformPreprocessing(preprocessed_image, preprocessed_mask);

    cv::Mat input_image;
    cv::Mat input_mask;
    cv::Mat input_annotation_mask;
    double rotated_angle;

    // prepare hog inputs
    PrepareInput(
        preprocessed_image,
        preprocessed_mask,
        annotation,
        training_scale_factor_,
        input_image,
        input_mask,
        input_annotation_mask,
        rotated_angle);

    // cv::imshow("sonar_source_image", sonar_source_image_);
    // cv::imshow("preprocessed_image", preprocessed_image);
    // cv::imshow("input_image", input_image);
    // cv::waitKey(15);

    cv::imshow("input_image", input_image);
    cv::imshow("input_mask", input_mask);
    cv::imshow("input_annotation_mask", input_annotation_mask);
    cv::waitKey();

    // validate positive input
    if (!positive_input_validate_ ||
        (positive_input_validate_ &&
        ValidatePositiveInput(input_mask, input_annotation_mask))) {
        // compute positive gradients
        ComputePositive(input_image, input_annotation_mask, gradient_positive);
    }


    // compute negative gradients
    ComputeNegative(input_image, input_mask, input_annotation_mask, gradient_negative);
}

void HOGDetector::CreateAnnotationMask(
     const cv::Size& size,
     const std::vector<cv::Point>& annotation,
     cv::Mat& annotation_mask)
{
    annotation_mask = cv::Mat::zeros(size, CV_8UC1);
    std::vector<std::vector<cv::Point> > contours;
    contours.push_back(annotation);
    cv::drawContours(annotation_mask, contours, -1, cv::Scalar(255), CV_FILLED);
}

void HOGDetector::OrientationNormalize(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    const cv::Mat& annotation_mask,
    cv::RotatedRect bbox,
    cv::Mat& rotated_image,
    cv::Mat& rotated_mask,
    cv::Mat& rotated_annotation_mask,
    double& rotated_angle)
{
    cv::Point2f center = cv::Point2f(source_image.cols/2, source_image.rows/2);
    rotated_angle = (bbox.size.width>=bbox.size.height) ? bbox.angle : bbox.angle+90;
    image_util::rotate(annotation_mask, rotated_annotation_mask, rotated_angle, center);
    RotateInput(source_image, source_mask, center, rotated_angle, rotated_image, rotated_mask);
}

void HOGDetector::RotateInput(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    const cv::Point2f& center,
    double angle,
    cv::Mat& rotated_image,
    cv::Mat& rotated_mask)
{
    image_util::rotate(source_image, rotated_image, angle, center);
    image_util::rotate(source_mask, rotated_mask, angle, center);
}

void HOGDetector::ComputePositive(
    const cv::Mat& source_image,
    const cv::Mat& annotation_mask,
    std::vector<cv::Mat>& gradient_list_positive)
{
    cv::Mat input_image;
    PreparePositiveInput(source_image, annotation_mask, input_image);

    if (show_positive_window_) {
        image_util::show_scale("positive_input_image", input_image, 1.5);
        cv::waitKey(15);
    }

    ComputeGradient(input_image, gradient_list_positive);
}

void HOGDetector::PreparePositiveInput(
    const cv::Mat& source_image,
    const cv::Mat& annotation_mask,
    cv::Mat& result_image)
{
    cv::Rect bounding_rect = image_util::get_bounding_rect(annotation_mask);
    result_image = source_image(bounding_rect);

    cv::resize(result_image, result_image, window_size_);
    result_image.convertTo(result_image, CV_8U, 255.0);
}

void HOGDetector::ComputeNegative(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    const cv::Mat& annotation_mask,
    std::vector<cv::Mat>& gradient_list_negative)
{

    cv::Mat input_image;
    cv::Mat input_mask;
    cv::Mat input_annotation_mask;

    PrepareNegativeInput(source_image, source_mask, annotation_mask,
        input_image, input_mask, input_annotation_mask);

    ComputeNegativeGradient(input_image, input_mask, input_annotation_mask, gradient_list_negative);
}

void HOGDetector::PrepareNegativeInput(
    const cv::Mat& source_image,
    const cv::Mat& source_mask,
    const cv::Mat& annotation_mask,
    cv::Mat& result_image,
    cv::Mat& result_mask,
    cv::Mat& result_annotation_mask)
{
    // copy region of interest
    cv::Rect bounding_rect = image_util::get_bounding_rect(source_mask);
    source_image(bounding_rect).copyTo(result_image);
    source_mask(bounding_rect).copyTo(result_mask);
    annotation_mask(bounding_rect).copyTo(result_annotation_mask);
    result_mask.setTo(0, result_annotation_mask);
    result_image.convertTo(result_image, CV_8U, 255.0);
}

void HOGDetector::ComputeNegativeGradient(
    const cv::Mat& src,
    const cv::Mat& mask,
    const cv::Mat& annotation_mask,
    std::vector<cv::Mat>& gradient_list_negative)
{
   cv::Size sz = src.size();
   cv::Size win = window_size_;

   int yy, xx;
   for (int y = 0; y < sz.height; y+=win.height) {

       if (((sz.height-y)/(float)sz.height) < 0.1) continue;

       yy = ((y+win.height)>=sz.height) ? sz.height-win.height : y;

       if (yy < 0) continue;

       for (int x = 0; x < sz.width; x+=win.width) {

           if (((sz.width-x)/(float)sz.width) < 0.1) continue;

           xx = ((x+win.width)>=sz.width) ? sz.width-win.width : x;

           if (xx < 0) continue;

           cv::Rect rc = cv::Rect(xx, yy, win.width, win.height);

           double m0 = cv::mean(mask(rc))[0]/255.0;
           double m1 = cv::mean(annotation_mask(rc))[0]/255.0;

           if (m0 > 0.5 && m1 < 0.2) {
               cv::Mat hog_input_negative;
               src(rc).copyTo(hog_input_negative);
               ComputeGradient(hog_input_negative, gradient_list_negative);
           }
       }
   }
}


void HOGDetector::ComputeGradient(
    const cv::Mat& source_image,
    std::vector<cv::Mat>& gradient_list)
{
    cv::Mat gray;
    std::vector<cv::Point> location;
    std::vector<float> descriptors;

    hog_descriptor_.compute(source_image, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);

    gradient_list.push_back(cv::Mat(descriptors).clone());

    if (show_descriptor_) {
        cv::Mat rgb;
        cv::cvtColor(source_image, rgb, CV_GRAY2BGR);
        cv::imshow("gradient", get_hogdescriptor_visu(rgb, descriptors, window_size_));
        cv::waitKey(200);
    }

}

void HOGDetector::PrepareTrainingData(
    const std::vector<cv::Mat>& positive,
    const std::vector<cv::Mat>& negative,
    std::vector<int>& labels,
    cv::Mat& training_data)
{
    assert(!positive.empty());
    assert(!negative.empty());

    labels.clear();
    labels.insert(labels.end(), positive.size(), +1);
    labels.insert(labels.end(), negative.size(), -1);

    std::vector<cv::Mat> train_samples;
    train_samples.insert(train_samples.end(), positive.begin(), positive.end());
    train_samples.insert(train_samples.end(), negative.begin(), negative.end());

    const int rows = (int)train_samples.size();
    const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);

    training_data = cv::Mat(rows, cols, CV_32FC1);

    cv::Mat train_sample;
    std::vector<cv::Mat>::const_iterator itr = train_samples.begin();
    std::vector<cv::Mat>::const_iterator end = train_samples.end();
    for(int i = 0 ; itr != end ; ++itr, ++i) {
        itr->copyTo(train_sample);
        CV_Assert(train_sample.cols == 1 || train_sample.rows == 1);

        if(train_sample.cols == 1) {
            cv::transpose(train_sample, train_sample);
        }

        train_sample.copyTo(training_data.row(i));
    }
}
void HOGDetector::SVMTrain(
    const std::vector<int>& labels,
    const cv::Mat& training_data,
    const std::string& training_filename)
{
    // Set up SVM's parameters
    cv::SVMParams params;
    params.coef0 = 0.0;
    params.degree = 3.0;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
    params.gamma = 0;
    params.kernel_type = CvSVM::LINEAR;
    params.nu = 0.5;
    params.p = 0.1;
    params.C = 0.01;
    params.svm_type = CvSVM::EPS_SVR;
    LinearSVM svm;
    svm.train(training_data, cv::Mat(labels), cv::Mat(), cv::Mat(), params);
    svm.save(training_filename.c_str());
}

void HOGDetector::TransformLocation(
    const std::vector<cv::Rect>& locations,
    double scale,
    double rotate,
    cv::Point translate,
    cv::Size source_size,
    std::vector<cv::RotatedRect>& rotated_locations)
{
    cv::Size size = cv::Size(source_size.width/scale, source_size.height/scale);
    cv::Point center = cv::Point(size.width/2, size.height/2);

    std::vector<cv::Rect>::const_iterator it;
    for(it = locations.begin(); it != locations.end(); ++it ) {
        cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
        cv::Rect rc;
        rc.x = (*it).x/scale+translate.x/scale;
        rc.y = (*it).y/scale+translate.y/scale;
        rc.width = (*it).width/scale;
        rc.height = (*it).height/scale;
        cv::rectangle(mask, rc, cv::Scalar(255), CV_FILLED);
        image_util::rotate(mask, mask, rotate, center, sonar_source_image_.size());
        std::vector<cv::Point> contours = preprocessing::find_biggest_contour(mask);
        cv::RotatedRect bbox = cv::minAreaRect(contours);
        if (bbox.size.width<bbox.size.height) {
            double aux = bbox.size.width;
            bbox.size.width = bbox.size.height;
            bbox.size.height = aux;
            bbox.angle = bbox.angle+90;
        }
        rotated_locations.push_back(bbox);
    }

}

void HOGDetector::ResizeAnnotationPoints(
    const std::vector<cv::Point>& source_points,
    std::vector<cv::Point>& result_points)
{
    image_util::resize_points(
        source_points,
        result_points,
        sonar_holder_.cart_width_factor(),
        sonar_holder_.cart_height_factor());
}

void HOGDetector::FilterLocationInsideMask(
    const std::vector<cv::Rect>& locations,
    const std::vector<double>& weights,
    std::vector<cv::Rect>& result_locations,
    std::vector<double>& result_weights,
    const cv::Mat& input,
    const cv::Mat& mask)
{
    cv::Mat mat = cv::Mat::zeros(input.size(), CV_8UC1);

    std::vector<cv::Rect> new_locations;
    std::vector<double> new_weights;
    for (size_t i = 0; i < locations.size(); i++) {
        mat.setTo(0);
        cv::rectangle(mat, locations[i], cv::Scalar(255), CV_FILLED);

        cv::Mat res;
        cv::bitwise_and(mat, mask, res);
        double area = (cv::sum(res)[0] / cv::sum(mat)[0]);

        if (area > 0.75) {
            new_locations.push_back(locations[i]);
            new_weights.push_back(weights[i]);
        }
    }
    result_locations = new_locations;
    result_weights = new_weights;
}

bool HOGDetector::ValidatePositiveInput(
    const cv::Mat& mask,
    const cv::Mat& annotation_mask)
{
    cv::Mat res;
    cv::bitwise_and(mask, annotation_mask, res);
    return (cv::sum(res)[0] / cv::sum(annotation_mask)[0]) > 0.7;
}

cv::Rect HOGDetector::GetLastDetectedBoundingRect(
    double scale,
    cv::Size max_size)
{
    const float PADDING_FACTOR = 0.2;

    cv::Rect bbox = last_detected_location_.boundingRect();
    bbox.x *= scale;
    bbox.y *= scale;
    bbox.width *= scale;
    bbox.height *= scale;

    int padx = bbox.width * PADDING_FACTOR;
    int pady = bbox.height * PADDING_FACTOR;

    bbox.x -= padx;
    bbox.y -= pady;
    bbox.width += padx*2;
    bbox.height  += pady*2;

    int w = max_size.width;
    int h = max_size.height;

    if (bbox.x < 0) bbox.x = 0;
    if (bbox.y < 0) bbox.y = 0;

    if (bbox.x+bbox.width >= w) {
        bbox.width = w-bbox.x-1;
    }

    if (bbox.y+bbox.height >= h) {
        bbox.height = h-bbox.y-1;
    }

    return bbox;
}

} /* namespace sonar_processing */
