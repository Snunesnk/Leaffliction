#include <iostream>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

int main(int argc, char* argv[]) {

	// Get the directory path from the command-line argument
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " -src <source_directory/source_image> -dst <destination_directory>" << std::endl;
		return 1;
	}
	std::string source;
	std::string destination;

	// Parse command-line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "-src" && i + 1 < argc) {
			source = argv[i + 1];
			++i;
		}
		else if (arg == "-dst" && i + 1 < argc) {
			destination = argv[i + 1];
			++i;
		}
		else if (arg == "-h") {
			// Display usage information and exit
			std::cout << "Usage: " << argv[0] << " -src <source_directory/source_image> -dst <destination_directory>" << std::endl;
			return 0;
		}
	}
#ifdef _MSC_VER
	source = "images/Apple_scab/Image (42).JPG";
	destination = "images/test2";
#endif
	// Check if source is a .JPG file
	if (source.length() >= 4 && source.substr(source.length() - 4) == ".JPG") {
		// Load an image from the specified file path
		cv::Mat image = cv::imread(source, cv::IMREAD_COLOR);
		if (image.empty()) {
			std::cerr << "Unable to load the image." << std::endl;
			return -1;
		}
		// Create a vector to store multiple copies of the loaded image
		std::vector<cv::Mat> images;
		for (int i = 0; i < 7; i++) {
			images.push_back(image.clone());
		}
		// Apply various image processing operations to different copies of the image
		ImageProcessing::Rotate(images[1], 5.0, 45.0);
		ImageProcessing::Blur(images[2], 2.5, 3.5);
		ImageProcessing::Contrast(images[3], 1.5, 2.5);
		ImageProcessing::Scale(images[4], 1.5, 2.5);
		ImageProcessing::Illumination(images[5], 30, 60);
		ImageProcessing::Projective(images[6], 15, 30);

		// Create a mosaic image from the processed images
		std::vector<std::string> labels = { "Original", "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
		ImageUtils::CreateImageMosaic(images, "Transformation", labels);
		
		// Create a mosaic image from the processed images
		images.erase(images.begin());
		labels.erase(labels.begin());
		// Check if source and destination are provided
		if (destination.empty() || !std::filesystem::is_directory(destination)) {
			ImageUtils::SaveImages(source, images, labels);
		}
		else {
			const size_t lastSlashPosition = source.find_last_of('/');
			const std::string name = source.substr(lastSlashPosition + 1);
			ImageUtils::SaveImages(destination + name, images, labels);
		}
		cv::waitKey(0);
	}
	else {
		// Check if source and destination are provided
		if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
			std::cerr << "Missing source or destination directory. Use -h for help." << std::endl;
			return 1;
		}
		if (source.back() != '/') {
			source += "/";
		}
		if (destination.back() != '/') {
			destination += "/";
		}
		std::vector<std::string> names = ImageUtils::GgetImagesInDirectory(source);
		for (auto name : names) {
			// Load an image from the specified file path
			cv::Mat image = cv::imread(source + name, cv::IMREAD_COLOR);
			if (image.empty()) {
				std::cerr << "Unable to load the image." << std::endl;
				return -1;
			}
			// Create a vector to store multiple copies of the loaded image
			std::vector<cv::Mat> images;
			for (int i = 0; i < 6; i++) {
				images.push_back(image.clone());
			}
			// Apply various image processing operations to different copies of the image
			ImageProcessing::Rotate(images[0], 5.0, 45.0);
			ImageProcessing::Blur(images[1], 2.5, 3.5);
			ImageProcessing::Contrast(images[2], 1.5, 2.5);
			ImageProcessing::Scale(images[3], 1.5, 2.5);
			ImageProcessing::Illumination(images[4], 30, 60);
			ImageProcessing::Projective(images[5], 15, 30);

			// Create a mosaic image from the processed images
			std::vector<std::string> labels = { "Original", "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
			ImageUtils::SaveImages(destination + name, images, labels);
		}
	}
	return 0;
}
