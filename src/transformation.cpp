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
			int hue = hsvPixel[0];
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
	//images
	//	| -- - Apple_Black_rot
	//	| '--- 620 files
	//	| -- - Apple_healthy
	//	| '--- 1640 files
	//	| -- - Apple_rust
	//	| '--- 275 files
	//	| -- - Apple_scab
	//	| '--- 629 files
	//	| -- - Grape_Black_rot
	//	| '--- 1178 files
	//	| -- - Grape_Esca
	//	| '--- 1382 files
	//	| -- - Grape_healthy
	//	| '--- 422 files
	//	| -- - Grape_spot
	//	| '--- 1075 files
	//	'--- 0 files
	source = "images/Grape_spot/";
	destination = "images/Grape_spot";
#endif
	// Check if source is a .JPG file
	if (source.length() >= 4 && source.substr(source.length() - 4) == ".JPG") {
		// Load an image from the specified file path
		cv::Mat originalImage = cv::imread(source, cv::IMREAD_COLOR);
		if (originalImage.empty()) {
			std::cerr << "Unable to load the image." << std::endl;
			return -1;
		}

		std::vector<cv::Mat> images;
		images.push_back(originalImage.clone());

		ImageProcessing::EqualizeValueHistogram(originalImage);
		ImageProcessing::EqualizeSaturationHistogram(originalImage);
		ImageProcessing::CutShape(originalImage);
		ImageProcessing::EqualizeValueHistogram(originalImage);
		ImageProcessing::EqualizeSaturationHistogram(originalImage);

		// Create a vector to store multiple copies of the processed image
		for (int i = 0; i < 6; i++) {
			images.push_back(originalImage.clone());
		}

		// Apply various image processing operations to different copies of the image
		images[1] = originalImage.clone();
		ImageProcessing::SimpleBinarization(images[1], 254);
		ImageProcessing::ExtractRedChannel(images[2]);
		ImageProcessing::ExtractGreenChannel(images[3]);
		ImageProcessing::ExtractBlueChannel(images[4]);
		ImageProcessing::ExtractSaturation(images[5], 128);
		ImageProcessing::ExtractValue(images[6], 128);

		// Create a mosaic image from the processed images
		std::vector<std::string> labels = { "Original", "Shape", "RedChannel", "GreenChannel", "BlueChannel", "Saturation", "Value" };
		ImageUtils::CreateImageMosaic(images, "Transformation", labels);

		cv::waitKey(1);

		// Calculate intensity proportions and generate the graph
		std::vector<std::vector<std::pair<int, double>>> proportions = calculateProportionInIntensityRanges(images[0]);
		generateGraphScript(proportions);

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
			cv::Mat originalImage = cv::imread(source + name, cv::IMREAD_COLOR);
			if (originalImage.empty()) {
				std::cerr << "Unable to load the image." << std::endl;
				return -1;
			}
			std::vector<cv::Mat> images;

			ImageProcessing::EqualizeValueHistogram(originalImage);
			ImageProcessing::EqualizeSaturationHistogram(originalImage);
			ImageProcessing::CutShape(originalImage);
			ImageProcessing::EqualizeValueHistogram(originalImage);
			ImageProcessing::EqualizeSaturationHistogram(originalImage);

			// Create a vector to store multiple copies of the loaded image
			for (int i = 0; i < 6; i++) {
				images.push_back(originalImage.clone());
			}

			// Apply various image processing operations to different copies of the image
			images[0] = originalImage.clone();
			ImageProcessing::SimpleBinarization(images[0], 254);
			ImageProcessing::ExtractRedChannel(images[1]);
			ImageProcessing::ExtractGreenChannel(images[2]);
			ImageProcessing::ExtractBlueChannel(images[3]);
			ImageProcessing::ExtractSaturation(images[4], 128);
			ImageProcessing::ExtractValue(images[5], 128);

			// Save the processed images with their respective labels
			std::vector<std::string> labels = { "Shape", "RedChannel", "GreenChannel", "BlueChannel", "Saturation", "Value" };
			ImageUtils::SaveImages(destination + name, images, labels);
		}
	}
	return 0;
}
