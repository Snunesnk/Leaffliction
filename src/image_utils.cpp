#include "image_utils.h"

void ImageUtils::CreateImageMosaic(const std::vector<cv::Mat> images, const std::string name, const std::vector<std::string>& labels) {
	constexpr double labelFontSize = 0.75;
	constexpr double labelThickness = 1;

	std::vector<cv::Mat> cols;
	for (int j = 0; j < images.size(); j++) {
		// Image
		const cv::Mat image = images[j];
		const std::string label = labels[j];
		cv::Mat labeledImage(image.rows + 30, image.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		image.copyTo(labeledImage(cv::Rect(0, 30, image.cols, image.rows)));
		// Label
		const int textWidth = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, labelFontSize, labelThickness, 0).width;
		const int xPos = (labeledImage.cols - textWidth) / 2;
		cv::putText(labeledImage, label, cv::Point(xPos, 20), cv::FONT_HERSHEY_SIMPLEX, labelFontSize, cv::Scalar(255, 255, 255), labelThickness);
		
		cols.push_back(labeledImage);
	}
	cv::Mat mosaic;
	cv::hconcat(cols, mosaic);
	cv::imshow(name, mosaic);
}

void ImageUtils::SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types) {
	const size_t lastSlashPosition = filePath.find_last_of('/');
	const size_t lastPointPosition = filePath.find_last_of('.');
	const std::string saveDirectory = filePath.substr(0, lastSlashPosition + 1);
	const std::string imageName = filePath.substr(lastSlashPosition + 1, lastPointPosition - lastSlashPosition - 1);

	for (int i = 0; i < images.size(); i++) {
		const std::string filename = saveDirectory + imageName + "_" + types[i] + ".JPG";
		cv::imwrite(filename, images[i]);
		std::cout << "Saved : " << filename << std::endl;
	}
}

std::vector<std::string> ImageUtils::GgetImagesInDirectory(const std::string& directoryPath) {
	std::vector<std::string> images;
	size_t imageCount = 0;
	for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
		std::filesystem::path entryPath = entry.path();
		if (std::filesystem::is_regular_file(entryPath)) {
			std::string fileName = entryPath.filename().generic_string();
			size_t pos = fileName.find('.');
			if (pos != std::string::npos) {
				std::string fileType = fileName.substr(pos, fileName.size());
				if (fileType == ".JPG") {
					images.push_back(fileName);
					imageCount++;
				}
			}
		}
	}
	std::cout << directoryPath << std::endl;
	std::cout << imageCount << " files" << std::endl;
	return images;
}