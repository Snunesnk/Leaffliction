#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

class ImageProcessing {
public:
    static void applyRotate(cv::Mat& image, double angle);
    static void applyBlur(cv::Mat& image, double sigma);
    static void applyContrast(cv::Mat& image, double alpha);
    static void applyScale(cv::Mat& image, double factor);
    static void applyIllumination(cv::Mat& image, int brightness);
    static void applyProjective(cv::Mat& image);
    static void applyConvertToGrayScale(cv::Mat& image);
    static void applyColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color);
    static void applyEqualizeHistogram(cv::Mat& image);
    static void applySimpleBinarization(cv::Mat& inputOutputImage, int threshold);
    static void applyDetectORBKeyPoints(cv::Mat& image);
};

#endif
