#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <filesystem>

class ImageUtils {
public:
	static void CreateImageMosaic(const std::vector<cv::Mat> images, const std::string name, const std::vector<std::string>& labels);
	static void SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types);
	static std::vector<std::string> GgetImagesInDirectory(const std::string& directoryPath);
};

#endif

