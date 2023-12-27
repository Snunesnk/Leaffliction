#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>

class ImageUtils {
public:
	static std::mutex mutex; // Mutex for thread-safe updates
	static int progress;
	static int numComplete;

	static void ShowMosaic(const std::vector<cv::Mat>& images, const std::string& name, const std::vector<std::string>& labels);
	static void SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types);
	static std::vector<std::string> GetImagesInDirectory(const std::string& directoryPath, int generation);
	static void SaveTFromToDirectory(const std::string& source, const std::string& destination, int generation);
	static void SaveAFromToDirectory(const std::string& source, const std::string& destination, int generation);
};

#endif

