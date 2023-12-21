#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

class ImageProcessing
{
public:
	static void Rotate(cv::Mat& image, double minDistr, double maxDistr);
	static void Distort(cv::Mat& image);
	static void Flip(cv::Mat& image);
	static void Shear(cv::Mat& image, double minDistr, double maxDistr);
	static void Scale(cv::Mat& image, double minDistr, double maxDistr);
	static void Projective(cv::Mat& image, float minDistr, float maxDistr);

	static void ConvertToGray(cv::Mat& inputImage);
	static void BinarizeImage(cv::Mat& inputImage);
	static void ExtractYChannel(cv::Mat& inputImage);
	static void ApplyCannyEdgeDetection(cv::Mat& inputImage);
	static void ApplyGaussianBlur(cv::Mat& inputImage, int kernelSize);
	static void ApplyContrastEnhancement(cv::Mat& inputImage, double factor);

	static void ColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color = { 255, 255, 255 });
	static void EqualizeHistogram(cv::Mat& image);
	static void EqualizeHistogramColor(cv::Mat& image);
	static void EqualizeHistogramSaturation(cv::Mat& image);
	static void EqualizeHistogramValue(cv::Mat& image);
	static void DetectORBKeyPoints(cv::Mat& image);

	static std::vector<cv::Point> GetConvexHullPoints(cv::Mat image);
	static void CropImageWithPoints(cv::Mat& image, const std::vector<cv::Point>& points);
	static double calculateAspectRatioOfObjects(cv::Mat image);

	static void CutLeaf(cv::Mat& image);
};

#endif
