#include "image_utils.h"
#include "image_processing.h"

#include <filesystem>


std::mutex ImageUtils::mutex;
int ImageUtils::progress;
int ImageUtils::numComplete;

void ImageUtils::SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types)
{
	const size_t lastSlashPos = filePath.find_last_of('/');
	const size_t lastPointPos = filePath.find_last_of('.');
	const std::string saveDir = filePath.substr(0, lastSlashPos + 1);
	const std::string imgName = filePath.substr(lastSlashPos + 1, lastPointPos - lastSlashPos - 1);

	for (int i = 0; i < images.size(); i++) {
		const std::string outputFilename = saveDir + imgName + "_" + types[i] + ".JPG";
		cv::imwrite(outputFilename, images[i]);
		{
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);
			std::cout << "\r\033[K" << "Saved : " << outputFilename << std::flush;
		}		
	}
}

std::vector<std::string> ImageUtils::GetImagesInDirectory(const std::string& directoryPath, int generation)
{
	std::vector<std::string> images;
	size_t imageCount = 0;

	for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
		if (entry.is_regular_file() && entry.path().extension() == ".JPG") {
			std::string fileName = entry.path().filename().generic_string();
			images.push_back(fileName);
			if (--generation == 0) {
				break;
			}
		}
	}

	std::sort(images.begin(), images.end(), [](const std::string& a, const std::string& b) {
		size_t posA = a.find(')');
		size_t posB = b.find(')');

		std::string numeroA = a.substr(a.find('(') + 1, posA - a.find('(') - 1);
		std::string numeroB = b.substr(b.find('(') + 1, posB - b.find('(') - 1);

		try {
			int intA = std::stoi(numeroA);
			int intB = std::stoi(numeroB);
			return intA < intB;
		}
		catch (...) {
			throw std::runtime_error("Error filename");
		}
		}
	);

	return images;
}

void ImageUtils::SaveTFromToDirectory(const std::string& source, const std::string& destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);

	for (auto i = 0; i < names.size(); i++) {

		// Load image
		cv::Mat originalImage = cv::imread(source + names[i]);
		if (originalImage.empty()) {
			throw std::runtime_error("Unable to load the image. " + source + names[i]);
		}

		// Process images
		std::vector<cv::Mat> images;
		cv::Mat clone = originalImage.clone();
		ImageProcessing::ExtractLeafAndRescale(clone);
		for (int i = 0; i < 6; i++) {
			images.push_back(clone.clone());
		}
		cv::GaussianBlur(images[1], images[1], {5, 5}, 0);
		ImageProcessing::EqualizeHistogramColor(images[2]);
		ImageProcessing::DetectORBKeyPoints(images[3]);
		ImageProcessing::EqualizeHistogramValue(images[4]);
		ImageProcessing::EqualizeHistogramSaturation(images[5]);

		// Save
		std::vector<std::string> transformations = { "T1", "T2", "T3", "T4", "T5", "T6" };
		ImageUtils::SaveImages(destination + names[i], images, transformations);

		{
			// Progression
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);
			int progress = (++ImageUtils::progress) * 100 / ImageUtils::numComplete;
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n" << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
			std::cout << "\033[A";
		}
	}
}

void ImageUtils::SaveAFromToDirectory(const std::string& source, const std::string& destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);
	for (auto i = 0; i < names.size(); i++) {
		// Load an image from the specified file path
		cv::Mat image = cv::imread(source + names[i]);
		if (image.empty()) {
			throw std::runtime_error("Unable to load the image. " + source + names[i]);
		}
		// Create a vector to store multiple copies of the loaded image
		std::vector<cv::Mat> images;
		for (int i = 0; i < 6; i++) {
			images.push_back(image.clone());
		}
		// Apply various image processing operations to different copies of the image
		ImageProcessing::Rotate(images[0], 5.0, 45.0);
		ImageProcessing::Distort(images[1]);
		ImageProcessing::Flip(images[2]);
		ImageProcessing::Shear(images[3], 0.2, 0.3);
		ImageProcessing::Scale(images[4], 0.5, 0.9);
		ImageProcessing::Projective(images[5], 30, 40);

		// Create a mosaic image from the processed images
		std::vector<std::string> labels = { "Rotate", "Distort", "Flip", "Shear", "Scale", "Projective" };
		ImageUtils::SaveImages(destination + names[i], images, labels);

		{
			// Progression
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);		
			int progress = (++ImageUtils::progress) * 100 / ImageUtils::numComplete;
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n" << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
			std::cout << "\033[A";
		}
	}
}