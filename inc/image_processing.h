#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

class ImageProcessing {
public:
	static void Rotate(cv::Mat& image, double minDistr, double maxDistr);
	static void Blur(cv::Mat& image, double minDistr, double maxDistr);
	static void Contrast(cv::Mat& image, double minDistr, double maxDistr);
	static void Scale(cv::Mat& image, double minDistr, double maxDistr);
	static void Illumination(cv::Mat& image, double minDistr, double maxDistr);
	static void Projective(cv::Mat& image, double minDistr, double maxDistr);
	static void ConvertToGrayScale(cv::Mat& image);
	static void ColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color = { 255, 255, 255 });
	static void EqualizeHistogram(cv::Mat& image);
	static void EqualizeHistogramColor(cv::Mat& image);
	static void EqualizeHistogramSaturation(cv::Mat& image);
	static void EqualizeHistogramValue(cv::Mat& image);
	static void SimpleBinarization(cv::Mat& inputOutputImage, int threshold);
	static void DetectORBKeyPoints(cv::Mat& image);
	static void ExtractHue(cv::Mat& image, double lower, double upper);
	static void ExtractSaturation(cv::Mat& image, double threshold);
	static void ExtractValue(cv::Mat& image, double threshold);
	static void ExtractRedChannel(cv::Mat& image);
	static void ExtractGreenChannel(cv::Mat& image);
	static void ExtractBlueChannel(cv::Mat& image);
	static void drawPolylinesAroundObject(cv::Mat image);
	static void drawRectangleAroundObject(cv::Mat image);
	static std::vector<cv::Point> GetConvexHullPoints(cv::Mat image);
	static std::vector<cv::Point> getMinimumBoundingRectanglePoints(cv::Mat image);
	static void CropImageWithPoints(cv::Mat& image, const std::vector<cv::Point>& points);
	static double calculateAspectRatioOfObjects(cv::Mat image);
	static void CutLeaf(cv::Mat& image);
};

#endif
