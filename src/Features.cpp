#include "sonar_processing/ImageUtils.hpp"
#include "sonar_processing/Features.hpp"

namespace sonar_processing {

void features::saliency_gray(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    cv::Mat src = src_arr.getMat();

    int height = src.size().height;
    int width = src.size().width;

    int minimum_dimension = std::min(width, height);

    int number_of_scale = 3;

    std::vector<int> N(number_of_scale);
    for (int i = 0; i < number_of_scale; i++){
        int scale = (1 << (i+1));
        N[i] = minimum_dimension/scale;
    }

    cv::Mat sm = cv::Mat::zeros(src.size(), CV_32FC1);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            if (!mask_arr.empty()) {
                cv::Mat mask = mask_arr.getMat();
                if (mask.at<uchar>(y, x) == 0) continue;
            }

            float val = src.at<float>(y, x);

            float cv_sum = 0;

            for (int k = 0; k < N.size(); k++) {
                int y1 = std::max(0, y-N[k]);
                int y2 = std::min(y+N[k], height-1);
                int x1 = std::max(0, x-N[k]);
                int x2 = std::min(x+N[k], width-1);

                cv::Rect rc = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

                float mean = cv::mean(src(rc))[0];
                float diff = val-mean;
                cv_sum += (diff*diff);
            }

            sm.at<float>(y, x) = cv_sum;
        }
    }
    
    sm.copyTo(dst_arr);
}

void features::saliency_color(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    cv::Mat src = src_arr.getMat();

    int height = src.size().height;
    int width = src.size().width;

    cv::Mat rgb;
    cv::cvtColor(src, rgb, CV_BGR2RGB);

    cv::Mat lab;
    image_utils::rgb2lab(rgb, lab);

    cv::Mat L, A, B;
    image_utils::split_channels(lab, L, A, B);

    int minimum_dimension = std::min(width, height);

    int number_of_scale = 3;

    std::vector<int> N(number_of_scale);
    for (int i = 0; i < number_of_scale; i++){
        int scale = (1 << (i+1));
        N[i] = minimum_dimension/scale;
    }

    cv::Mat sm = cv::Mat::zeros(src.size(), CV_32F);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            if (!mask_arr.empty()) {
                cv::Mat mask = mask_arr.getMat();
                if (mask.at<uchar>(y, x) == 0) continue;
            }

            float l_val = L.at<float>(y, x);
            float a_val = A.at<float>(y, x);
            float b_val = B.at<float>(y, x);
            
            float cv_sum = 0;
            for (int k = 0; k < N.size(); k++) {
                int y1 = std::max(0, y-N[k]);
                int y2 = std::min(y+N[k], height-1);
                int x1 = std::max(0, x-N[k]);
                int x2 = std::min(x+N[k], width-1);

                cv::Rect rc = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));

                float l_mean = cv::mean(L(rc))[0];
                float a_mean = cv::mean(A(rc))[0];
                float b_mean = cv::mean(B(rc))[0];

                float l_diff = pow(l_val-l_mean, 2);
                float a_diff = pow(a_val-a_mean, 2);
                float b_diff = pow(b_val-b_mean, 2);

                cv_sum += (l_diff + a_diff + b_diff);
            }

            sm.at<float>(y, x) = cv_sum;
        }
    }
    sm.copyTo(dst_arr);
}
    
void features::saliency(cv::InputArray src_arr, cv::OutputArray dst_arr) {
    saliency(src_arr, dst_arr, cv::noArray());
}

void features::saliency(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    if (src_arr.channels() == 3) {
        saliency_color(src_arr, dst_arr, mask_arr);
    }
    else {
        saliency_gray(src_arr, dst_arr, mask_arr);
    };
}

} /* namespace sonar_processing */
