#include "ImageUtil.hpp"
#include "ShapeProcessing.hpp"
#include "Utils.hpp"
#include "ImageFiltering.hpp"

namespace sonar_processing {

namespace image_filtering {

void saliency_gray(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
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
    
    cv::Mat integral;
    cv::integral(src, integral, CV_32F);

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

                int NN = (x2-x1)*(y2-y1);
                float mean = image_util::integral_image_sum<float>(integral, x1, y1, x2, y2) / NN;
                float diff = val-mean;
                cv_sum += (diff*diff);
            }

            sm.at<float>(y, x) = cv_sum;
        }
    }
    
    sm.copyTo(dst_arr);
}

void saliency_color(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    cv::Mat src = src_arr.getMat();

    int height = src.size().height;
    int width = src.size().width;

    cv::Mat rgb;
    cv::cvtColor(src, rgb, CV_BGR2RGB);

    cv::Mat lab;
    image_util::rgb2lab(rgb, lab);

    cv::Mat L, A, B;
    image_util::split_channels(lab, L, A, B);

    int minimum_dimension = std::min(width, height);

    int number_of_scale = 3;

    std::vector<int> N(number_of_scale);
    for (int i = 0; i < number_of_scale; i++){
        int scale = (1 << (i+1));
        N[i] = minimum_dimension/scale;
    }

    cv::Mat sm = cv::Mat::zeros(src.size(), CV_32F);
    cv::Mat mask = mask_arr.getMat();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            if (!mask_arr.empty() && mask.at<uchar>(y, x) == 0) continue;

            float l_val = L.at<float>(y, x);
            float a_val = A.at<float>(y, x);
            float b_val = B.at<float>(y, x);
            
            float cv_sum = 0;
        
            for (int k = 0; k < N.size(); k++) {
                int y1 = std::max(0, y-N[k]);
                int y2 = std::min(y+N[k], height-1);
                int x1 = std::max(0, x-N[k]);
                int x2 = std::min(x+N[k], width-1);

                if (!mask_arr.empty() && mask.at<uchar>(y1, x1) == 0) break;
                if (!mask_arr.empty() && mask.at<uchar>(y2, x2) == 0) break;

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

void border_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    cv::Mat G;
    cv::Mat Gx, Gy;
    cv::Mat Gx2, Gy2;

    cv::Sobel( src_arr, Gx, CV_16S, 1, 0, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gx, Gx2 );

    cv::Sobel( src_arr, Gy, CV_16S, 0, 1, CV_SCHARR, 0.5, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( Gy, Gy2 );

    cv::addWeighted(Gx2, 0.5, Gy2, 0.5, 0, G);

    G.copyTo(dst_arr, mask_arr);
}

void saliency_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::InputArray mask_arr) {
    if (src_arr.channels() == 3) {
        saliency_color(src_arr, dst_arr, mask_arr);
    }
    else {
        saliency_gray(src_arr, dst_arr, mask_arr);
    }
}

void mean_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr) {
    CV_Assert(src_arr.type() == CV_32FC1);
    cv::Mat integral;
    cv::integral(src_arr, integral, CV_32F);
    integral_mean_filter(integral, dst_arr, ksize, mask_arr);
}

void integral_mean_filter(cv::InputArray integral_arr, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr) {
    CV_Assert(integral_arr.type() == CV_32FC1);

    int w = integral_arr.size().width-1;
    int h = integral_arr.size().height-1;

    cv::Mat mask = mask_arr.getMat();
    cv::Mat integral = integral_arr.getMat();
    cv::Mat dst = cv::Mat::zeros(cv::Size(w, h), CV_32FC1);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            cv::Rect r = utils::neighborhood_rect(x, y, ksize, w, h);

            if (!mask.empty() && mask.at<uchar>(y, x) == 0){
                continue;
            }

            dst.at<float>(y, x) = image_util::integral_image_sum<float>(integral, r) / (float)r.area();
        }
    }
    
    dst.copyTo(dst_arr);
}

void meand_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, int ksize_outer, int ksize_inner, cv::InputArray mask_arr) {
    CV_Assert(src_arr.type() == CV_32FC1);

    cv::Mat src = src_arr.getMat();

    cv::Mat integral;
    cv::integral(src, integral, CV_32F);

    cv::Mat mask = mask_arr.getMat();

    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Rect r0 = utils::neighborhood_rect(x, y, ksize_outer, src.cols, src.rows);
            cv::Rect r1 = utils::neighborhood_rect(x, y, ksize_inner, src.cols, src.rows);
            float m0 = image_util::integral_image_sum<float>(integral, r0) / (float)r0.area();
            float m1 = image_util::integral_image_sum<float>(integral, r1) / (float)r1.area();

            dst.at<float>(y, x) = m1-m0;
        }
    }
}

void mean_difference_filter(cv::InputArray src_arr0, cv::InputArray src_arr1, cv::OutputArray dst_arr, int ksize, cv::InputArray mask_arr) {
    CV_Assert(src_arr0.type() == src_arr1.type());
    CV_Assert(src_arr0.size() == src_arr1.size());
    CV_Assert(src_arr0.type() == CV_32FC1);

    cv::Mat integral;
    cv::integral(src_arr0, integral, CV_32F);
    
    int w = src_arr0.size().width;
    int h = src_arr0.size().height;

    cv::Mat mask = mask_arr.getMat();
    cv::Mat src1 = src_arr1.getMat();
    cv::Mat dst = cv::Mat::zeros(src_arr0.size(), src_arr0.type());

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            cv::Rect r = utils::neighborhood_rect(x, y, ksize, w, h);

            if (!mask.empty() && mask.at<uchar>(y, x) == 0) {
                continue;
            }

            float m = image_util::integral_image_sum<float>(integral, r)/(float)r.area();
            float d = src1.at<float>(y, x)-m;
            dst.at<float>(y, x) = (d < 0) ? 0 : ((d > 1) ? 1 : d);
        }
    }
    
    dst.copyTo(dst_arr);
}

// void minimum_blob_filter(cv::InputArray src_arr, cv::OutputArray dst_arr, cv::Size min_size, int mode, bool convex_hull) {
//     cv::Mat src = src_arr.getMat();
//     dst_arr.create(src.size(), CV_8UC1);
// 
//     std::vector<std::vector<cv::Point> > contours = shape_processing::find_contours(src, min_size, mode, convex_hull);
// 
//     if (!contours.empty()) {
//         cv::Mat dst = dst_arr.getMat();
//         dst.setTo(0);
//         cv::drawContours(dst, contours, -1, cv::Scalar(255), CV_FILLED);
//     }
// }

void saliency_mapping(cv::InputArray src_arr, cv::OutputArray dst_arr, int block_count, cv::InputArray mask_arr) {
    CV_Assert(src_arr.type() == CV_32FC1);

    cv::Mat src = src_arr.getMat();

    int width = src.cols;
    int height = src.rows;
    int block_width = width/block_count;
    int block_height = height/block_count;
    int rwidth = block_width*block_count;
    int rheight = block_height*block_count;

    cv::Mat res = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat cnt = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat mask = mask_arr.getMat();
    
    cv::Mat mask_32f;
    mask.convertTo(mask_32f, CV_32F, 1.0/255.0);

    cv::Mat integral;
    cv::integral(src, integral, CV_32F);
        
    int number_of_rows=(height-block_height)/(block_height/2)+1;
    int number_of_cols=(width-block_width)/(block_width/2)+1;

    int total_blocks = number_of_rows*number_of_cols;

    std::vector<cv::Rect> rects(total_blocks); 
    std::vector<float> means(total_blocks);

    int idx = 0;
    for (int i = 0; i < height-block_height; i+=block_height/2) {
        for (int j = 0; j < width-block_width; j+=block_width/2) {
            cv::Rect rc = cv::Rect(j, i, block_width, block_height);
            rects[idx] = rc;
            means[idx] = image_util::integral_image_sum<float>(integral, rc)/(float)rc.area();
            idx++;
        }
    }

    for (int k = 0; k < total_blocks-1; k++) {
        float m0 = means[k];
        for (int l = k+1; l < total_blocks; l++) {
            float m1 = means[l];
            cv::Rect rc = rects[l];
            res(rc) += fabs(m1-m0);
            cnt(rc) += 1.0;
        }
    }

    res /= cnt;
    res.copyTo(dst_arr);
}

} /* namespace image_filtering */

} /* namespace  sonar_processing */
