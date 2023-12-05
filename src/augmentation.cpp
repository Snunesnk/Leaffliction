#include <iostream>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

void saveFromToDirectory(std::string& source, std::string& destination, int generation) {
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}
	if (source.back() != '/') {
		source += "/";
	}
	if (destination.back() != '/') {
		destination += "/";
	}
	std::vector<std::string> names = ImageUtils::GgetImagesInDirectory(source);
	for (auto i = 0; i < names.size(); i++) {
		// Load an image from the specified file path
		cv::Mat image = cv::imread(source + names[i], cv::IMREAD_COLOR);
		if (image.empty()) {
			throw std::runtime_error("Unable to load the image.");
		}
		// Create a vector to store multiple copies of the loaded image
		std::vector<cv::Mat> images;
		for (int i = 0; i < 6; i++) {
			images.push_back(image.clone());
		}
		// Apply various image processing operations to different copies of the image
		ImageProcessing::Rotate(images[0], 5.0, 45.0);
		ImageProcessing::Blur(images[1], 1.5, 2.0);
		ImageProcessing::Contrast(images[2], 1.5, 2.0);
		ImageProcessing::Scale(images[3], 1.25, 1.5);
		ImageProcessing::Illumination(images[4], 30, 40);
		ImageProcessing::Projective(images[5], 30, 40);

		// Create a mosaic image from the processed images
		std::vector<std::string> labels = { "Original", "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
		ImageUtils::SaveImages(destination + names[i], images, labels);

		// Progression
		int progress = (i + 1) * 100 / names.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";
		if ((generation -= 6) <= 0) {
			std::cout << "\nGeneration ended at -gen parameter." << std::endl;
			break;
		}
	}
	std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush;
}

void displayMosaic(const std::string& source, std::string& destination) {
	// Load an image from the specified file path
	cv::Mat image = cv::imread(source, cv::IMREAD_COLOR);
	if (image.empty()) {
		throw std::runtime_error("Unable to load the image.");
	}
	// Create a vector to store multiple copies of the loaded image
	std::vector<cv::Mat> images;
	for (int i = 0; i < 7; i++) {
		images.push_back(image.clone());
	}
	// Apply various image processing operations to different copies of the image
	ImageProcessing::Rotate(images[1], 5.0, 45.0);
	ImageProcessing::Blur(images[2], 1.5, 2.0);
	ImageProcessing::Contrast(images[3], 1.5, 2.0);
	ImageProcessing::Scale(images[4], 1.25, 1.5);
	ImageProcessing::Illumination(images[5], 30, 40);
	ImageProcessing::Projective(images[6], 30, 40);

	// Create a mosaic image from the processed images
	std::vector<std::string> labels = { "Original", "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
	ImageUtils::CreateImageMosaic(images, "Transformation", labels);
	cv::waitKey(0);
}

int main(int argc, char* argv[]) {
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " -src <source_directory/source_image> -dst <destination_directory> -gen <generation_max>");
		}
		std::string source;
		std::string destination;
		int generation = std::numeric_limits<int>::max();

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
			else if (arg == "-gen" && i + 1 < argc) {
				try {
					generation = std::atoi(argv[i + 1]);
				}
				catch (...) {
					throw std::runtime_error("Unable to read the -gen value.");
				}
				++i;
			}
			else if (arg == "-h") {
				std::cout << "Usage: " << argv[0] << " -src <source_directory/source_image> -dst <destination_directory> -gen <generation_max>" << std::endl;
				return 0;
			}
		}
#ifdef _MSC_VER
		//images
		//	|--- Apple_Black_rot
		//	|    '--- 620 files
		//	|--- Apple_healthy
		//	|    '--- 1640 files
		//	|--- Apple_rust
		//	|    '--- 275 files
		//	|--- Apple_scab
		//	|    '--- 629 files
		//	|--- Grape_Black_rot
		//	|    '--- 1178 files
		//	|--- Grape_Esca
		//	|    '--- 1382 files
		//	|--- Grape_healthy
		//	|    '--- 422 files
		//	|--- Grape_spot
		//	|    '--- 1075 files
		//	'--- 0 files
		source = "images/Grape_spot";
		destination = "images/Grape_spot";
		generation = 1640 - 1075;
#endif
		// Check if source is a .JPG file
		if (source.length() >= 4 && source.substr(source.length() - 4) == ".JPG") {
			displayMosaic(source, destination);
		}
		else {
			saveFromToDirectory(source, destination, generation);
		}
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Use -h for help." << std::endl;
		return 1;
	}
	return 0;
}
