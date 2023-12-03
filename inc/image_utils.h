#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <filesystem>

class ImageUtils
{
public:
	static void CreateImageMosaic(std::vector<cv::Mat> images, std::string name);
	static void SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types);
};

#endif

