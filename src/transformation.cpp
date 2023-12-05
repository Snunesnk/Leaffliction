#include <iostream>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

// Function to calculate the proportion of pixels in an intensity range for each channel
std::vector<std::vector<std::pair<int, double>>> calculateProportionInIntensityRanges(const cv::Mat& image) {
	int channelCount = 6;
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	int intensityRanges = 256;
	std::vector<std::vector<int>> intensityCounts(channelCount, std::vector<int>(intensityRanges, 0));
	int totalPixels = hsvImage.rows * hsvImage.cols;
	for (int y = 0; y < hsvImage.rows; ++y) {
		for (int x = 0; x < hsvImage.cols; ++x) {
			cv::Vec3b hsvPixel = hsvImage.at<cv::Vec3b>(y, x);
			// Extract values from each channel
			int hue = static_cast<int>(hsvPixel[0] * 1.417); // adapt openCV hue : 180.0 * 1.417 = 255.06
			int saturation = hsvPixel[1];
			int value = hsvPixel[2];
			// Increment the intensity count for each channel
			intensityCounts[0][hue]++;
			intensityCounts[1][saturation]++;
			intensityCounts[2][value]++;
			// Extract values from Blue, Green and Red channels (BGR)
			int blue = image.at<cv::Vec3b>(y, x)[0];
			int green = image.at<cv::Vec3b>(y, x)[1];
			int red = image.at<cv::Vec3b>(y, x)[2];
			intensityCounts[3][blue]++;
			intensityCounts[4][green]++;
			intensityCounts[5][red]++;
		}
	}
	std::vector<std::vector<std::pair<int, double>>> intensityProportions(channelCount);
	// Calculate proportions for each channel
	for (int i = 0; i < channelCount; ++i) {
		for (int j = 0; j < intensityRanges; ++j) {
			double proportion = (static_cast<double>(intensityCounts[i][j]) / totalPixels) * 100.0;
			intensityProportions[i].push_back(std::make_pair(j, proportion));
		}
	}
	return intensityProportions;
}

// Function to create and execute the Python script for generating the graph
void generateGraphScript(const std::vector<std::vector<std::pair<int, double>>>& proportions) {
	// Create and open a text file for writing
	std::ofstream pythonScript("script.py");
	if (!pythonScript.is_open()) {
		throw std::runtime_error("Erreur : Impossible de créer le fichier Python (script.py).");
	}
	std::vector<std::string> channels = { "hue", "saturation", "value", "blue", "green", "red" };
	std::vector<std::string> colors = { "purple", "cyan", "orange", "blue", "green", "red" };
	// Python script
	pythonScript << "import matplotlib.pyplot as plt\n";
	pythonScript << "import numpy as np\n";
	pythonScript << "plt.figure(figsize=(9, 7))\n";
	// Iterate through the proportions and add data for each channel
	for (int i = 0; i < proportions.size(); ++i) {
		pythonScript << "data" << i << " = np.array([";
		for (const auto& pair : proportions[i]) {
			pythonScript << "[" << pair.first << "," << pair.second << "],";
		}
		pythonScript << "])\n";
		pythonScript << "intensities" << i << " = data" << i << "[:, 0]\n";
		pythonScript << "proportion_values" << i << " = data" << i << "[:, 1]\n";
		pythonScript << "plt.plot(intensities" << i << ", proportion_values" << i << ", label='" << channels[i] << "', color='" << colors[i] << "', alpha=0.5)\n";
	}
	pythonScript << "plt.legend(loc='upper right')\n";
	pythonScript << "plt.xlabel('Pixel intensity')\n";
	pythonScript << "plt.ylabel('Proportion of pixels (%)')\n";
	pythonScript << "plt.show()";
	pythonScript.close();

	if (system("python script.py &") != 0) {
		throw std::runtime_error("Error: Failed to execute the Python command.");
	}
}

void SaveTransformationsFromToDirectory(std::string& source, std::string& destination, int generation) {
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
		cv::Mat originalImage = cv::imread(source + names[i], cv::IMREAD_COLOR);
		if (originalImage.empty()) {
			throw std::runtime_error("Unable to load the image.");
		}
		std::vector<cv::Mat> images;

		// Create a vector to store multiple copies of the loaded image
		for (int i = 0; i < 6; i++) {
			images.push_back(originalImage.clone());
		}

		// Apply various image processing operations to different copies of the image
		std::vector<cv::Point> points = ImageProcessing::ExtractShape(images[0]);
		for (int i = 0; i < 6; i++) {
			ImageProcessing::cropImageWithPoints(images[i], points);
		}
		ImageProcessing::EqualizeHistogramColor(images[1]);
		ImageProcessing::EqualizeHistogramSaturation(images[2]);
		ImageProcessing::EqualizeHistogramValue(images[3]);
		ImageProcessing::ConvertToGrayScale(images[4]);
		ImageProcessing::EqualizeHistogram(images[5]);

		// Save the processed images with their respective labels
		std::vector<std::string> labels = { "T1", "T2", "T3", "T4", "T5", "T6" };
		ImageUtils::SaveImages(destination + names[i], images, labels);

		// Progression
		int progress = (i + 1) * 100 / names.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";
		if ((generation -= 6) <= 0) {
			std::cout << "\nGeneration ended at -gen parameter." << std::endl;
			return;
		}
		break;
	}
	std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush;
}

void displayImageTranformations(const std::string& source) {
	// Load an image from the specified file path
	cv::Mat originalImage = cv::imread(source, cv::IMREAD_COLOR);
	if (originalImage.empty()) {
		throw std::runtime_error("Unable to load the image.");
	}

	std::vector<cv::Mat> images;
	images.push_back(originalImage.clone());


	// Create a vector to store multiple copies of the processed image
	for (int i = 0; i < 6; i++) {
		images.push_back(originalImage.clone());
	}

	// Apply various image processing operations to different copies of the image
	std::vector<cv::Point> points = ImageProcessing::ExtractShape(images[1]);
	for (int i = 1; i < 7; i++) {
		ImageProcessing::cropImageWithPoints(images[i], points);
	}
	ImageProcessing::EqualizeHistogramColor(images[2]);
	ImageProcessing::EqualizeHistogramSaturation(images[3]);
	ImageProcessing::EqualizeHistogramValue(images[4]);
	ImageProcessing::ConvertToGrayScale(images[5]);
	ImageProcessing::EqualizeHistogram(images[6]);

	// Save the processed images with their respective labels
	std::vector<std::string> labels = { "Original", "T1-Shape", "T2-HEColor", "T3-HESaturation", "T4-HEValue", "T5-GrayScale", "T6-HEGrayScale" };
	ImageUtils::CreateImageMosaic(images, "Transformation", labels);

	cv::waitKey(1);

	// Calculate intensity proportions and generate the graph
	std::vector<std::vector<std::pair<int, double>>> proportions = calculateProportionInIntensityRanges(images[0]);
	generateGraphScript(proportions);

	cv::waitKey(0);
}

int main(int argc, char* argv[]) {
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " -src <source_directory/source_image> -dst <destination_directory> [-gen <generation_max>]");

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
				std::cout << "Usage: " << argv[0] << " -src <source_directory/source_image> -dst <destination_directory>  [-gen <generation_max>]" << std::endl;
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
		source = "images/Grape_Esca/";
		destination = "images/test";
#endif

		// Check if source is a .JPG file
		if (source.length() >= 4 && source.substr(source.length() - 4) == ".JPG") {
			displayImageTranformations(source);
		}
		else {
			SaveTransformationsFromToDirectory(source, destination, generation);
		}
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Use -h for help." << std::endl;
		return 1;
	}
	return 0;
}
