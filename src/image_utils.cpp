#include "image_utils.h"
#include "image_processing.h"

#include <filesystem>

std::mutex ImageUtils::mutex;
int ImageUtils::progress;
int ImageUtils::numComplete;

void ImageUtils::ShowMosaic(const std::vector<cv::Mat> &images, const std::string &name, const std::vector<std::string> &targets)
{
	const double FontSize = 0.75;
	const double Thickness = 1;

	std::vector<cv::Mat> cols;
	for (size_t j = 0; j < images.size(); j++)
	{

		const cv::Mat image = images[j];
		const std::string target = targets[j];
		cv::Mat mosaicImage(image.rows + 30, image.cols, image.type(), cv::Scalar(0, 0, 0));
		image.copyTo(mosaicImage(cv::Rect(0, 30, image.cols, image.rows)));

		const int textWidth = cv::getTextSize(target, cv::FONT_HERSHEY_SIMPLEX, FontSize, Thickness, 0).width;
		const int xPos = (mosaicImage.cols - textWidth) / 2;
		cv::putText(mosaicImage, target, cv::Point(xPos, 20), cv::FONT_HERSHEY_SIMPLEX, FontSize, cv::Scalar(255, 255, 255), Thickness);

		cols.push_back(mosaicImage);
	}
	cv::Mat mosaic;
	cv::hconcat(cols, mosaic);
	cv::imshow(name, mosaic);
}

void ImageUtils::SaveImages(const std::string &filePath, const std::vector<cv::Mat> &images, const std::vector<std::string> &types)
{
	const size_t lastSlashPos = filePath.find_last_of('/');
	const size_t lastPointPos = filePath.find_last_of('.');
	const std::string saveDir = filePath.substr(0, lastSlashPos + 1);
	const std::string imgName = filePath.substr(lastSlashPos + 1, lastPointPos - lastSlashPos - 1);

	for (size_t i = 0; i < images.size(); i++)
	{
		const std::string outputFilename = saveDir + imgName + "_" + types[i] + ".JPG";
		cv::imwrite(outputFilename, images[i]);
		{
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);
			std::cout << "\r\033[K"
					  << "Saved : " << outputFilename << std::flush;
		}
	}
}

std::vector<std::string> ImageUtils::GetImagesInDirectory(const std::string &directoryPath, int generation)
{
	std::vector<std::string> images;

	for (const auto &entry : std::filesystem::directory_iterator(directoryPath))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".JPG")
		{
			std::string fileName = entry.path().filename().generic_string();
			images.push_back(fileName);
			if (--generation == 0)
			{
				break;
			}
		}
	}

	std::sort(images.begin(), images.end(), [](const std::string &a, const std::string &b)
			  {
		size_t posA = a.find(')');
		size_t posB = b.find(')');
		std::string numeroA = a.substr(a.find('(') + 1, posA - a.find('(') - 1);
		std::string numeroB = b.substr(b.find('(') + 1, posB - b.find('(') - 1);
		int intA = std::stoi(numeroA);
		int intB = std::stoi(numeroB);
		return intA < intB; });

	return images;
}

void ImageUtils::SaveTFromToDirectory(const std::string &source, const std::string &destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination))
	{
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);

	for (size_t i = 0; i < names.size(); i++)
	{

		// Load image
		cv::Mat originalImage = cv::imread(source + names[i]);
		if (originalImage.empty())
		{
			throw std::runtime_error("Unable to load the image. " + source + names[i]);
		}

		// Process images
		std::vector<cv::Mat> images;
		cv::Mat clone = originalImage.clone();
		ImageProcessing::ExtractLeafAndRescale(clone);
		for (int i = 0; i < 6; i++)
		{
			images.push_back(clone.clone());
		}
		cv::GaussianBlur(images[1], images[1], {5, 5}, 0);
		ImageProcessing::EqualizeHistogramColor(images[2]);
		ImageProcessing::DetectORBKeyPoints(images[3]);
		ImageProcessing::EqualizeHistogramValue(images[4]);
		ImageProcessing::EqualizeHistogramSaturation(images[5]);

		// Save
		std::vector<std::string> transformations = {"T1", "T2", "T3", "T4", "T5", "T6"};
		ImageUtils::SaveImages(destination + names[i], images, transformations);

		{
			// Progression
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);
			int progress = (++ImageUtils::progress) * 100 / ImageUtils::numComplete;
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n"
					  << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
			std::cout << "\033[A";
		}
	}
}

void ImageUtils::SaveAFromToDirectory(const std::string &source, const std::string &destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination))
	{
		throw std::runtime_error("Missing source or destination directory.");
	}

	int progress = 0;
	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation / 6 + 1);
	for (size_t i = 0; i < names.size(); i++)
	{

		// Load an image from the specified file path
		cv::Mat image = cv::imread(source + names[i]);
		if (image.empty())
		{
			throw std::runtime_error("Unable to load the image. " + source + names[i]);
		}
		std::vector<cv::Mat> images;

		// Apply various image processing operations to different copies of the image
		if (progress + 6 <= generation)
		{
			for (int i = 0; i < 6; i++)
			{
				images.push_back(image.clone());
			}
			ImageProcessing::Rotate(images[0], 5.0, 45.0);
			ImageProcessing::Distort(images[1]);
			ImageProcessing::Flip(images[2]);
			ImageProcessing::Shear(images[3], 0.2, 0.3);
			ImageProcessing::Scale(images[4], 0.5, 0.9);
			ImageProcessing::Projective(images[5], 30, 40);
			progress += 6;
		}
		else
		{
			if (progress == generation)
			{
				break;
			}
			size_t num = generation - progress;
			for (size_t i = 0; i < num; i++)
			{
				images.push_back(image.clone());
			}
			ImageProcessing::Rotate(images[0], 5.0, 45.0);
			if (num > 1)
			{
				ImageProcessing::Distort(images[1]);
			}
			if (num > 2)
			{
				ImageProcessing::Flip(images[2]);
			}
			if (num > 3)
			{
				ImageProcessing::Shear(images[3], 0.2, 0.3);
			}
			if (num > 4)
			{
				ImageProcessing::Scale(images[4], 0.5, 0.9);
			}
			progress += num;
		}

		// Create a mosaic image from the processed images
		std::vector<std::string> augmentations = {"Rotate", "Distort", "Flip", "Shear", "Scale", "Projective"};
		ImageUtils::SaveImages(destination + names[i], images, augmentations);

		// Progression
		{
			std::lock_guard<std::mutex> lock(ImageUtils::mutex);
			int progress = ((ImageUtils::progress += 6) * 100) / ImageUtils::numComplete;
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n"
					  << "[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
			std::cout << "\033[A";
		}
	}
}