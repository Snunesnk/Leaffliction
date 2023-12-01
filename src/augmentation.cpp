#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "image_processing.h"

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
		return 1;
	}
	std::string filePath = argv[1];
	filePath += "/Apple_Black_rot/image (56).JPG";
	cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
	if (image.empty())
	{
		std::cerr << "Unable to load the image." << std::endl;
		return -1;
	}
	cv::Mat applyRotateImage = image.clone();
	cv::Mat applyBlurImage = image.clone();
	cv::Mat applyContrastImage = image.clone();
	cv::Mat applyScaleImage = image.clone();
	cv::Mat applyIlluminationImage = image.clone();
	cv::Mat applyProjectiveImage = image.clone();
	ImageProcessing::applyRotate(applyRotateImage, 10.0);
	ImageProcessing::applyBlur(applyBlurImage, 3.0);
	ImageProcessing::applyContrast(applyContrastImage, 1.5);
	ImageProcessing::applyScale(applyScaleImage, 1.5);
	ImageProcessing::applyIllumination(applyIlluminationImage, 30);
	ImageProcessing::applyProjective(applyProjectiveImage);
	cv::imshow("image", image);
	cv::imshow("applyRotateImage", applyRotateImage);
	cv::imshow("applyBlurImage", applyBlurImage);
	cv::imshow("applyContrastImage", applyContrastImage);
	cv::imshow("applyScaleImage", applyScaleImage);
	cv::imshow("applyIlluminationImage", applyIlluminationImage);
	cv::imshow("applyProjectiveImage", applyProjectiveImage);
	cv::waitKey(0);
	return 0;
}
