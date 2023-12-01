#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <random>

void ImageProcessing::applyRotate(cv::Mat& image, double angle)
{
    // Calculate the size of the resulting image after rotation
    cv::Rect boundingRect = cv::RotatedRect(cv::Point2f(image.cols / 2.0, image.rows / 2.0), image.size(), angle).boundingRect();
    // Automatically calculate the scaling factor
    double scale_factor = std::min(static_cast<double>(image.cols) / boundingRect.width, static_cast<double>(image.rows) / boundingRect.height);
    // Calculate the rotation matrix with scaling
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(cv::Point2f(image.cols / 2.0, image.rows / 2.0), angle, scale_factor);
    // Create a new image with a white background of the size of the original image
    cv::Mat rotatedImage = cv::Mat::zeros(image.size(), image.type());
    // Apply rotation (and scaling) to the original image
    cv::warpAffine(image, rotatedImage, rotationMatrix, rotatedImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    // Replace the original image with the rotated and scaled image
    image = rotatedImage;
}

void ImageProcessing::applyBlur(cv::Mat& image, double sigma)
{
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void ImageProcessing::applyContrast(cv::Mat& image, double alpha)
{
    image.convertTo(image, -1, alpha, 0);
}

void ImageProcessing::applyScale(cv::Mat& image, double factor)
{
    cv::Point2f center(image.cols / 2.0, image.rows / 2.0);
    cv::Mat zoomMatrix = cv::getRotationMatrix2D(center, 0.0, factor);
    // Apply zoom to the original image
    cv::warpAffine(image, image, zoomMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void ImageProcessing::applyIllumination(cv::Mat& image, int brightness)
{
    image += cv::Scalar(brightness, brightness, brightness);
}

void ImageProcessing::applyProjective(cv::Mat& image)
{
    // Randomizer
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, 30);
    int topLY = distr(gen);
    int topLX = distr(gen);
    int topRY = distr(gen);
    int topRX = distr(gen);
    int botLY = distr(gen);
    int botLX = distr(gen);
    int botRY = distr(gen);
    int botRX = distr(gen);
    // Source points
    std::vector<cv::Point2f> srcPoints;
    srcPoints.push_back(cv::Point2f(0, 0));                           // Top-left corner
    srcPoints.push_back(cv::Point2f(image.cols - 1, 0));              // Top-right corner
    srcPoints.push_back(cv::Point2f(0, image.rows - 1));              // Bottom-left corner
    srcPoints.push_back(cv::Point2f(image.cols - 1, image.rows - 1)); // Bottom-right corner
    // Destination points
    std::vector<cv::Point2f> dstPoints;
    dstPoints.push_back(cv::Point2f(topLY, topLX));
    dstPoints.push_back(cv::Point2f(image.cols - topRY, topRX));
    dstPoints.push_back(cv::Point2f(botLY, image.rows - botLX));
    dstPoints.push_back(cv::Point2f(image.cols - botRY, image.rows - botRX));
    // Get the Perspective Transform Matrix i.e. M
    cv::Mat warpMatrix = cv::getPerspectiveTransform(srcPoints, dstPoints);
    // Apply the perspective transformation to the image
    cv::warpPerspective(image, image, warpMatrix, image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
}

void ImageProcessing::applyConvertToGrayScale(cv::Mat& image)
{
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

void ImageProcessing::applyColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color)
{
    // Convert the image to the HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
    // Create a mask for the specified color range
    cv::Mat colorMask;
    cv::inRange(hsvImage, lowerBound, upperBound, colorMask);
    // Set the color for the parts of the image that match the mask
    image.setTo(color, colorMask);
}

void ImageProcessing::applyEqualizeHistogram(cv::Mat& image)
{
    // Convert the image to grayscale
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    // Equalize the histogram
    cv::equalizeHist(image, image);
}

void ImageProcessing::applySimpleBinarization(cv::Mat& inputOutputImage, int threshold)
{
    // Ensure the image is in grayscale
    if (inputOutputImage.channels() > 1) {
        cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_BGR2GRAY);
    }
    // Apply binarization
    cv::threshold(inputOutputImage, inputOutputImage, threshold, 255, cv::THRESH_BINARY);
}

void ImageProcessing::applyDetectORBKeyPoints(cv::Mat& image)
{
    // Create an ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(image, keypoints);
    // Draw keypoints on the image
    cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}
