#include "image_processing.h"
#include "image_utils.h"

#include <iostream>
#include <filesystem>

void display(const std::string &source)
{

	// Load image
	cv::Mat image = cv::imread(source);
	if (image.empty())
	{
		throw std::runtime_error("Unable to load the image.");
	}

	// Process images
	std::vector<cv::Mat> images;
	for (int i = 0; i < 7; i++)
	{
		images.push_back(image.clone());
	}
	ImageProcessing::Rotate(images[1], 5.0, 45.0);
	ImageProcessing::Distort(images[2]);
	ImageProcessing::Flip(images[3]);
	ImageProcessing::Shear(images[4], 0.2, 0.3);
	ImageProcessing::Scale(images[5], 0.5, 0.9);
	ImageProcessing::Projective(images[6], 30, 40);

	// Show mosaic
	std::vector<std::string> augmentations = {"Original", "Rotate", "Distort", "Flip", "Shear", "Scale", "Projective"};
	ImageUtils::ShowMosaic(images, "Augmentation", augmentations);
	cv::waitKey(0);
}

void augmentation(const std::string &source, const std::string &destination, int generation)
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
		for (int i = 0; i < 6; i++)
		{
			images.push_back(originalImage.clone());
		}
		ImageProcessing::Rotate(images[0], 5.0, 45.0);
		ImageProcessing::Distort(images[1]);
		ImageProcessing::Flip(images[2]);
		ImageProcessing::Shear(images[3], 0.2, 0.3);
		ImageProcessing::Scale(images[4], 0.5, 0.9);
		ImageProcessing::Projective(images[5], 30, 40);

		// Save the processed images
		std::vector<std::string> augmentations = {"Rotate", "Distort", "Flip", "Shear", "Scale", "Projective"};
		const std::string destinationPath = destination + imageNames[i];
		const size_t lastSlashPos = destinationPath.find_last_of('/');
		const size_t lastPointPos = destinationPath.find_last_of('.');
		const std::string saveDir = destinationPath.substr(0, lastSlashPos + 1);
		const std::string imgName = destinationPath.substr(lastSlashPos + 1, lastPointPos - lastSlashPos - 1);
		for (size_t i = 0; i < images.size(); i++)
		{
			const std::string outputFilename = saveDir + imgName + "_" + augmentations[i] + ".JPG";
			cv::imwrite(outputFilename, images[i], {cv::IMWRITE_JPEG_QUALITY, 100});
			std::cout << "\r\033[K"
					  << "Saved : " << outputFilename << std::flush;
		}

		// Progression
		int progress = (i + 1) * 100 / imageNames.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n"
				  << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
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
		std::string destination = "images/augmented_directory/";
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
		if (source.length() >= 4 && source.substr(source.length() - 4) == ".JPG")
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
			augmentation(source, destination, generation);
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
