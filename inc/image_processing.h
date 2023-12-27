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
	static void EqualizeHistogramColor(cv::Mat& image);
	static void EqualizeHistogramSaturation(cv::Mat& image);
	static void EqualizeHistogramValue(cv::Mat& image);
	static void DetectORBKeyPoints(cv::Mat& image);

	static void ExtractLeafAndRescale(cv::Mat& image);

	static std::vector<double> ExtractTextureCaracteristics(const cv::Mat& image);
	static std::vector<double> ExtractColorCaracteristics(const cv::Mat& image);

private :
	static cv::Mat CalculateGLCM(const cv::Mat& img);
	static std::vector<double> ExtractGLCMFeatures(const cv::Mat& glcm);
};

#endif
