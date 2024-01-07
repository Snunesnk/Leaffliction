#include "image_processing.h"
#include "image_utils.h"

#include <iostream>
#include <fstream>
#include <filesystem>

std::vector<std::vector<std::pair<int, double>>> calculateProportionInIntensityRanges(const cv::Mat &image)
{
	int channelCount = 6;
	cv::Mat hsvImage;
	cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
	int intensityRanges = 256;
	std::vector<std::vector<int>> intensityCounts(channelCount, std::vector<int>(intensityRanges, 0));
	int totalPixels = hsvImage.rows * hsvImage.cols;
	for (int y = 0; y < hsvImage.rows; ++y)
	{
		for (int x = 0; x < hsvImage.cols; ++x)
		{
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
	for (int i = 0; i < channelCount; ++i)
	{
		for (int j = 0; j < intensityRanges; ++j)
		{
			double proportion = (static_cast<double>(intensityCounts[i][j]) / totalPixels) * 100.0;
			intensityProportions[i].push_back(std::make_pair(j, proportion));
		}
	}
	return intensityProportions;
}

void generateGraphScript(const std::vector<std::vector<std::pair<int, double>>> &proportions)
{
	// Create python file
	std::ofstream pythonScript("script.py");
	if (!pythonScript.is_open())
	{
		throw std::runtime_error("Erreur : Impossible de créer le fichier Python (script.py).");
	}
	std::vector<std::string> channels = {"hue", "saturation", "value", "blue", "green", "red"};
	std::vector<std::string> colors = {"purple", "cyan", "orange", "blue", "green", "red"};

	// Python script
	pythonScript << "import matplotlib.pyplot as plt\n";
	pythonScript << "import numpy as np\n";
	pythonScript << "plt.figure(figsize=(9, 7))\n";

	// Iterate through the proportions and add data for each channel
	for (int i = 2; i < 3; ++i)
	{
		pythonScript << "data" << i << " = np.array([";
		for (const auto &pair : proportions[i])
		{
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

	if (system("python script.py") != 0)
	{
		throw std::runtime_error("Failed to execute the Python command.");
	}
}

void display(const std::string &source)
{
	// Load image
	cv::Mat originalImage = cv::imread(source);
	if (originalImage.empty())
	{
		throw std::runtime_error("Unable to load the image.");
	}

	// Process images
	std::vector<cv::Mat> images;
	images.push_back(originalImage.clone());
	ImageProcessing::ExtractLeafAndRescale(originalImage);
	for (int i = 0; i < 6; i++)
	{
		images.push_back(originalImage.clone());
	}
	cv::GaussianBlur(images[2], images[2], {5, 5}, 0);
	ImageProcessing::EqualizeHistogramColor(images[3]);
	ImageProcessing::DetectORBKeyPoints(images[4]);
	ImageProcessing::EqualizeHistogramValue(images[5]);
	ImageProcessing::EqualizeHistogramSaturation(images[6]);

	// Show the processed images
	std::vector<std::string> transformations = {"Original", "T1", "T2", "T3", "T4", "T5", "T6"};
	ImageUtils::ShowMosaic(images, "Transformation", transformations);
	cv::waitKey(1);

	// Calculate intensity proportions and generate the graph
	std::vector<std::vector<std::pair<int, double>>> proportions = calculateProportionInIntensityRanges(images[0]);
	generateGraphScript(proportions);
	cv::waitKey(0);
}

void transformation(const std::string &source, const std::string &destination, int generation)
{
	// Check if destination already exists
	if (std::filesystem::exists(destination))
	{
		if (!std::filesystem::is_directory(destination))
		{
			throw std::runtime_error("Destination already exists, but it's not a directory.");
		}
	}
	else
	{
		std::filesystem::create_directory(destination);
	}

	// Process
	std::vector<std::string> imageNames = ImageUtils::GetImagesInDirectory(source, generation);
	for (size_t i = 0; i < imageNames.size(); i++)
	{

		// Load image
		cv::Mat originalImage = cv::imread(source + imageNames[i]);
		if (originalImage.empty())
		{
			throw std::runtime_error("Unable to load the image.");
		}

		// Process images
		std::vector<cv::Mat> images;
		ImageProcessing::ExtractLeafAndRescale(originalImage);
		for (int i = 0; i < 6; i++)
		{
			images.push_back(originalImage.clone());
		}
		cv::GaussianBlur(images[1], images[1], {5, 5}, 0);
		ImageProcessing::EqualizeHistogramColor(images[2]);
		ImageProcessing::DetectORBKeyPoints(images[3]);
		ImageProcessing::EqualizeHistogramValue(images[4]);
		ImageProcessing::EqualizeHistogramSaturation(images[5]);

		// Save the processed images
		std::vector<std::string> transformations = {"T1", "T2", "T3", "T4", "T5", "T6"};
		const std::string destinationPath = destination + imageNames[i];
		const size_t lastSlashPos = destinationPath.find_last_of('/');
		const size_t lastPointPos = destinationPath.find_last_of('.');
		const std::string saveDir = destinationPath.substr(0, lastSlashPos + 1);
		const std::string imgName = destinationPath.substr(lastSlashPos + 1, lastPointPos - lastSlashPos - 1);
		for (size_t i = 0; i < images.size(); i++)
		{
			const std::string outputFilename = saveDir + imgName + "_" + transformations[i] + ".JPG";
			cv::imwrite(outputFilename, images[i], {cv::IMWRITE_JPEG_QUALITY, 100});
			std::cout << "\r\033[K"
					  << "Saved : " << outputFilename << std::flush;
		}

		// Progression
		int progress = (i + 1) * 100 / imageNames.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";
	}
	std::cout << "\n\r\033[K\033[A\r\033[K";
}

int main(int argc, char *argv[])
{
	try
	{
		if (argc < 2)
		{
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " <source_path> -dst <destination_path> -gen <num_generations>");
		}
		// Apple_Black_rot     620 files
		// Apple_healthy       1640 files
		// Apple_rust          275 files
		// Apple_scab          629 files
		// Grape_Black_rot     1178 files
		// Grape_Esca          1382 files
		// Grape_healthy       422 files
		// Grape_spot          1075 files
		std::string source = argv[1];
		std::string destination = "images/transformed_directory/";
		int generation = 1640;

		// Parse command-line arguments
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			if (arg == "-dst" && i + 1 < argc)
			{
				destination = argv[i + 1];
				++i;
			}
			else if (arg == "-gen" && i + 1 < argc)
			{
				generation = std::atoi(argv[i + 1]);
				if (generation > 1640)
				{
					generation = 1640;
				}
				++i;
			}
			else if (arg == "-h")
			{
				std::cout << "Usage: " << argv[0] << " <source_path> -dst <destination_path> -gen <num_generations>" << std::endl;
				return 0;
			}
		}
		if (source.length() > 4 && source.substr(source.length() - 4) == ".JPG")
		{
			display(source);
		}
		else
		{
			if (source.back() != '/')
			{
				source += "/";
			}
			if (destination.back() != '/')
			{
				destination += "/";
			}
			transformation(source, destination, generation);
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
