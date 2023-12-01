#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

// Converts the image to grayscale
void applyConvertToGrayScale(cv::Mat& image) {
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
}

// Applies a color filter for segmentation
void applyColorFiltering(cv::Mat& image, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Scalar color) {
	// Convert the image to the HSV color space
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

	// Create a mask for the specified color range
	cv::Mat colorMask;
	cv::inRange(hsvImage, lowerBound, upperBound, colorMask);

	// Set the color for the parts of the image that match the mask
	image.setTo(color, colorMask);
}

// Equalizes the histogram of the grayscale image
void applyEqualizeHistogram(cv::Mat& image) {
	// Convert the image to grayscale
	cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

	// Equalize the histogram
	cv::equalizeHist(image, image);
}

// Applies adaptive thresholding to the grayscale image
void applyBinarisationSimple(cv::Mat& inputOutputImage, int seuil) {
	// Assurez-vous que l'image est en niveaux de gris (échelle de gris)
	if (inputOutputImage.channels() > 1) {
		cv::cvtColor(inputOutputImage, inputOutputImage, cv::COLOR_BGR2GRAY);
	}

	// Appliquer la binarisation
	cv::threshold(inputOutputImage, inputOutputImage, seuil, 255, cv::THRESH_BINARY);
}

// Detects keypoints with ORB descriptors and draws them on the image
void applyDetectORBKeyPoints(cv::Mat& image) {
	// Create an ORB detector
	cv::Ptr<cv::ORB> orb = cv::ORB::create();

	// Detect keypoints
	std::vector<cv::KeyPoint> keypoints;
	orb->detect(image, keypoints);

	// Draw keypoints on the image
	cv::drawKeypoints(image, keypoints, image, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
}

// Applique un filtre en fonction de la luminosité des pixels
void applyBrightnessFilter(cv::Mat& image, double seuil) {
	// Assurez-vous que l'image est en niveaux de gris (échelle de gris)
	if (image.channels() > 1) {
		cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
	}

	// Parcourez l'image et remplacez les pixels lumineux par du blanc
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			if (image.at<uchar>(y, x) > seuil) {
				image.at<uchar>(y, x) = 255; // Remplacez par du blanc (255)
			}
		}
	}
}

// Function to calculate the proportion of pixels in an intensity range for each channel
std::vector<std::vector<std::pair<int, double>>> calculateProportionInIntensityRanges(const cv::Mat& image) {
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

	int intensityRanges = 256; // 256 intensity levels (0-255)

	std::vector<std::vector<int>> intensityCounts(7, std::vector<int>(intensityRanges, 0)); // Seven vectors for seven channels

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
			int blue = image.at<cv::Vec3b>(y, x)[0]; // Blue channel
			int green = image.at<cv::Vec3b>(y, x)[1]; // Green channel
			int red = image.at<cv::Vec3b>(y, x)[2]; // Red channel

			intensityCounts[3][blue]++;
			intensityCounts[4][green]++;
			intensityCounts[5][red]++;

			// Extract value for luminosity (average of RGB)
			int luminosity = (green + red + blue) / 3;

			intensityCounts[6][luminosity]++;
		}
	}

	std::vector<std::vector<std::pair<int, double>>> intensityProportions(7); // Seven vectors for seven channels

	// Calculate proportions for each channel
	for (int i = 0; i < 7; ++i) {
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

	// Verify if the Python file was successfully created
	if (!pythonScript.is_open()) {
		throw std::runtime_error("Erreur : Impossible de créer le fichier Python (script.py).");
	}

	std::vector<std::string> channels = { "hue", "saturation", "value", "blue", "green", "red", "lightness" };
	std::vector<std::string> colors = { "purple", "cyan", "orange", "blue", "green", "red", "gray" };

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

	// Execute the Python command
	if (system("python script.py &") != 0) {
		throw std::runtime_error("Error: Failed to execute the Python command.");
	}
}


int main(int argc, char* argv[]) {
	try {
		if (argc != 2) {
			std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
			return 1;
		}

		std::string filePath = argv[1];
		filePath += "/Apple_Black_rot/image (1).JPG";

		// Load an image from a JPEG file
		cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

		// Check if the image was loaded successfully
		if (image.empty()) {
			std::cerr << "Unable to load the image." << std::endl;
			return -1;
		}

		cv::Mat applyConvertToGrayScaleImage = image.clone();
		cv::Mat applyGreenFilteringImage = image.clone();
		cv::Mat applyEqualizeHistogramImage = image.clone();
		cv::Mat applyBinarisationSimpleImage = image.clone();
		cv::Mat applyDetectORBKeyPointsImage = image.clone();
		cv::Mat applyRedFilteringRedImage = image.clone();
		cv::Mat applyBrightnessFilterImage = image.clone();


		applyConvertToGrayScale(applyConvertToGrayScaleImage);
		cv::Scalar lowerGreen = cv::Scalar(21, 0, 0); // Low HSV range for greens
		cv::Scalar upperGreen = cv::Scalar(90, 255, 255); // High HSV range for greens
		applyColorFiltering(applyGreenFilteringImage, lowerGreen, upperGreen, cv::Scalar(0, 255, 0));
		applyEqualizeHistogram(applyEqualizeHistogramImage);
		applyBinarisationSimple(applyBinarisationSimpleImage, 128);
		applyDetectORBKeyPoints(applyDetectORBKeyPointsImage);
		cv::Scalar lowerRed = cv::Scalar(0, 50, 50); // Low HSV range for reds
		cv::Scalar upperRed = cv::Scalar(20, 255, 255); // High HSV range for reds
		applyColorFiltering(applyRedFilteringRedImage, lowerRed, upperRed, cv::Scalar(0, 0, 255));
		double alpha = 0.5; // Facteur d'éclaircissement
		double beta = 30;  // Valeur à ajouter à chaque pixel
		applyBrightnessFilter(applyBrightnessFilterImage, 128);

		cv::imshow("image", image);
		cv::imshow("applyConvertToGrayScaleImage", applyConvertToGrayScaleImage);
		cv::imshow("applyGreenFilteringImage", applyGreenFilteringImage);
		cv::imshow("applyEqualizeHistogramImage", applyEqualizeHistogramImage);
		cv::imshow("applyBinarisationSimpleImage", applyBinarisationSimpleImage);
		cv::imshow("applyDetectORBKeyPointsImage", applyDetectORBKeyPointsImage);
		cv::imshow("applyRedFilteringRedImage", applyRedFilteringRedImage);
		cv::imshow("applyBrightnessFilterImage", applyBrightnessFilterImage);

		cv::waitKey(1);

		// Call the function to calculate intensity proportions
		std::vector<std::vector<std::pair<int, double>>> proportions = calculateProportionInIntensityRanges(image);
		// Call the function generate the graph
		generateGraphScript(proportions);

		cv::waitKey(0);
	}
	catch (const std::exception& e) {
		// Gérer les exceptions et afficher les messages d'erreur
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
