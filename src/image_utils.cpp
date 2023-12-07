#include "image_utils.h"
#include "image_processing.h"

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
	const size_t lastSlashPos = filePath.find_last_of('/');
	const size_t lastPointPos = filePath.find_last_of('.');
	const std::string saveDir = filePath.substr(0, lastSlashPos + 1);
	const std::string imgName = filePath.substr(lastSlashPos + 1, lastPointPos - lastSlashPos - 1);

	for (int i = 0; i < images.size(); i++) {
		const std::string outputFilename = saveDir + imgName + "_" + types[i] + ".JPG";
		cv::imwrite(outputFilename, images[i], { cv::IMWRITE_JPEG_QUALITY, 100 });
		std::cout << "\r\033[K" << "Saved : " << outputFilename;
	}
}

std::vector<std::string> ImageUtils::GetImagesInDirectory(const std::string& directoryPath, int generation) {
	std::vector<std::string> images;
	size_t imageCount = 0;

	for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
		if (entry.is_regular_file() && entry.path().extension() == ".JPG") {
			std::string fileName = entry.path().filename().generic_string();
			images.push_back(fileName);

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

	if (generation >= 0 && generation < images.size()) {
		images.erase(images.begin() + generation, images.end());
	}

	std::cout << directoryPath << std::endl;
	std::cout << images.size() << " files" << std::endl;
	return images;
}

void ImageUtils::SaveTFromToDirectory(std::string& source, std::string& destination, int generation) {
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}
	for (const auto& entry : std::filesystem::directory_iterator(destination)) {
		if (std::filesystem::is_regular_file(entry)) {
			std::string nomFichier = entry.path().filename().string();
			if (nomFichier.find('T') != std::string::npos) {
				std::filesystem::remove(entry.path());
				std::cout << "Deleted file : " << entry.path() << "\r\033[K";
			}
		}
	}
	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);
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
		//images[1] = originalImage.clone();
		// Plage pour le vert
		cv::Scalar green_lower(40, 0, 0); // Basse limite (H, S, V)
		cv::Scalar green_upper(100, 255, 255); // Haute limite (H, S, V)
		ImageProcessing::ColorFiltering(images[2], green_lower, green_upper, cv::Scalar(255, 0, 0));
		// Plage pour le rouge (rouge pur)
		cv::Scalar red_lower1(0, 0, 0); // Basse limite (H, S, V)
		cv::Scalar red_upper1(40, 255, 255); // Haute limite (H, S, V)
		ImageProcessing::ColorFiltering(images[3], red_lower1, red_upper1, cv::Scalar(255, 0, 0));


		//ImageProcessing::EqualizeHistogramSaturation(images[5]);
		//ImageProcessing::EqualizeHistogramValue(images[5]);

		for (int i = 0; i < 6; i++) {
			//ImageProcessing::CropImageWithPoints(images[i], points);
		}

		// Save the processed images with their respective labels
		std::vector<std::string> transformations = { "T1", "T2", "T3", "T4", "T5", "T6" };

		ImageUtils::SaveImages(destination + names[i], images, transformations);
		// Progression
		int progress = (i + 1) * 100 / names.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";

	}
	std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush << std::endl << "\r\033[K";
}

void ImageUtils::SaveAFromToDirectory(std::string& source, std::string& destination, int generation) {
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);
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
		std::vector<std::string> augmentations = { "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
		ImageUtils::SaveImages(destination + names[i], images, augmentations);

		// Progression
		int progress = (i * 6 + 1) * 100 / names.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";
		if ((generation -= 6) <= 0) {
			break;
		}
	}
	std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush << std::endl << "\r\033[K";
}