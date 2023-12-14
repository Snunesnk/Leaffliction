#include "image_utils.h"
#include "image_processing.h"

void ImageUtils::ShowMosaic(const std::vector<cv::Mat>& images, const std::string& name, const std::vector<std::string>& labels)
{
	const double labelFontSize = 0.75;
	const double labelThickness = 1;

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

void ImageUtils::SaveImages(const std::string& filePath, const std::vector<cv::Mat>& images, const std::vector<std::string>& types)
{
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

std::vector<std::string> ImageUtils::GetImagesInDirectory(const std::string& directoryPath, int generation)
{
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

double ImageUtils::Mean(const std::vector<double>& data)
{
	double sum = 0.0;
	double count = 0.0;
	for (const auto& value : data) {
		if (!std::isnan(value) && !std::isinf(value)) {
			sum += value;
			count++;
		}
	}
	if (count == 0) {
		return 0;
	}
	return sum / count;
}

double ImageUtils::StandardDeviation(const std::vector<double>& data)
{
	double m = Mean(data);
	double variance = 0.0;
	double count = 0;
	for (const auto& value : data) {
		if (!std::isnan(value) && !std::isinf(value)) {
			variance += std::pow(value - m, 2);
			count++;
		}
	}
	if (count == 0) {
		return 0;
	}
	return std::sqrt(variance / count);
}
void ImageUtils::SaveTFromToDirectory(std::string& source, std::string& destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);
	for (auto i = 0; i < names.size(); i++) {
		// Load an image from the specified file path
		cv::Mat originalImage = cv::imread(source + names[i], cv::IMREAD_COLOR);
		if (originalImage.empty()) {
			throw std::runtime_error("Unable to load the image. " + source + names[i]);
		}
		//ImageProcessing::EqualizeHistogramSaturation(originalImage);
		std::vector<cv::Mat> images;
		// Create a vector to store multiple copies of the loaded image
		for (int i = 0; i < 6; i++) {
			images.push_back(originalImage.clone());
		}

		// Apply various image processing operations to different copies of the image
		std::vector<cv::Point> pts = ImageProcessing::ExtractShape(images[0]);
		// Plage pour le vert
		cv::Scalar green_lower(40, 0, 0); // Basse limite (H, S, V)
		cv::Scalar green_upper(100, 255, 255); // Haute limite (H, S, V)
		ImageProcessing::ColorFiltering(images[2], green_lower, green_upper, cv::Scalar(255, 0, 0));
		ImageProcessing::CropImageWithPoints(images[2], pts);
		// Plage pour le rouge (rouge pur)
		cv::Scalar red_lower1(0, 0, 0); // Basse limite (H, S, V)
		cv::Scalar red_upper1(40, 255, 255); // Haute limite (H, S, V)
		ImageProcessing::ColorFiltering(images[3], red_lower1, red_upper1, cv::Scalar(255, 0, 0));
		ImageProcessing::CropImageWithPoints(images[3], pts);
		ImageProcessing::EqualizeHistogramSaturation(images[4]);


		//ImageProcessing::EqualizeHistogramValue(images[5]);
		cv::Mat hsvImage;
		cv::cvtColor(images[5], hsvImage, cv::COLOR_BGR2HSV_FULL);
		for (int i = 0; i < images[5].rows; i++) {
			for (int j = 0; j < images[5].cols; j++) {
				//hsvImage.at<cv::Vec3b>(i, j)[0] = 128;
				hsvImage.at<cv::Vec3b>(i, j)[2] = 128;
			}
		}
		cv::cvtColor(hsvImage, images[5], cv::COLOR_HSV2BGR_FULL);
		for (int i = 0; i < images[5].rows; i++) {
			for (int j = 0; j < images[5].cols; j++) {
				double min = images[5].at<cv::Vec3b>(i, j)[0];
				if (images[5].at<cv::Vec3b>(i, j)[2] < min) {
					min = images[5].at<cv::Vec3b>(i, j)[2];
				}
				images[5].at<cv::Vec3b>(i, j)[1] = (images[5].at<cv::Vec3b>(i, j)[1] < min ? 0 : images[5].at<cv::Vec3b>(i, j)[1] - min);
				images[5].at<cv::Vec3b>(i, j)[0] = (images[5].at<cv::Vec3b>(i, j)[0] < min ? 0 : images[5].at<cv::Vec3b>(i, j)[0] - min);
				images[5].at<cv::Vec3b>(i, j)[2] = (images[5].at<cv::Vec3b>(i, j)[2] < min ? 0 : images[5].at<cv::Vec3b>(i, j)[2] - min);
			}
		}


		ImageProcessing::Contrast(images[5], 2, 2);
		cv::GaussianBlur(images[5], images[5], { 11,11 }, 0);

		cv::cvtColor(images[5], hsvImage, cv::COLOR_BGR2HSV_FULL);
		std::vector<cv::Point> points;

		int tSize = 4;
		int startX = 0; // La position de départ en x
		int endX = images[5].cols - 1; // La position de fin en x
		int startY = 0; // La position de départ en y
		int endY = images[5].rows - 1; // La position de fin en y

		// Parcours de gauche à droite
		for (int i = startY; i <= endY; i++) {
			double save = 0.0;
			std::vector<double> stds;
			for (int k = 0; k < tSize; k++) {
				stds.push_back(hsvImage.at<cv::Vec3b>(i, k)[2]);
			}
			save = Mean(stds);
			for (int j = startX; j <= endX; j++) {
				stds.clear();
				for (int k = j; k < j + tSize && k <= endX; k++) {
					stds.push_back(hsvImage.at<cv::Vec3b>(i, k)[2]);
				}
				double result = Mean(stds);
				if (abs(result - save) > 5) {

					for (int k = j; k < j + tSize * 2 && k <= endX; k++) {
						images[5].at<cv::Vec3b>(i, k)[0] = 0;
						images[5].at<cv::Vec3b>(i, k)[1] = 0;
						images[5].at<cv::Vec3b>(i, k)[2] = 0;

					}
					points.push_back({ j + tSize * 2, i });
					break;
				}
				else {
					images[5].at<cv::Vec3b>(i, j)[0] = 0;
					images[5].at<cv::Vec3b>(i, j)[1] = 0;
					images[5].at<cv::Vec3b>(i, j)[2] = 0;
					save = (result + save) / 2;
				}
			}
		}
		// Parcours de droite à gauche
		for (int i = startY; i <= endY; i++) {
			double save = 0.0;
			std::vector<double> stds;
			for (int k = 0; k < tSize; k++) {
				stds.push_back(hsvImage.at<cv::Vec3b>(i, endX - k)[2]);
			}
			save = Mean(stds);
			for (int j = endX; j >= startX; j--) {
				stds.clear();
				for (int k = j - tSize + 1; k <= j && k >= startX; k++) {
					stds.push_back(hsvImage.at<cv::Vec3b>(i, k)[2]);
				}
				double result = Mean(stds);
				if (abs(result - save) > 5) {

					for (int k = j; k > j - tSize * 2 && k >= startX; k--) {
						images[5].at<cv::Vec3b>(i, k)[0] = 0;
						images[5].at<cv::Vec3b>(i, k)[1] = 0;
						images[5].at<cv::Vec3b>(i, k)[2] = 0;
					}
					points.push_back({ j - tSize * 2, i });
					break;
				}
				else {
					images[5].at<cv::Vec3b>(i, j)[0] = 0;
					images[5].at<cv::Vec3b>(i, j)[1] = 0;
					images[5].at<cv::Vec3b>(i, j)[2] = 0;
					save = (result + save) / 2;
				}
			}
		}
		// Parcours de haut en bas
		for (int j = startX; j <= endX; j++) {
			double save = 0.0;
			std::vector<double> stds;
			for (int k = 0; k < tSize; k++) {
				stds.push_back(hsvImage.at<cv::Vec3b>(endY - k, j)[2]);
			}
			save = Mean(stds);
			for (int i = endY; i >= startY; i--) {

				stds.clear();
				for (int k = i - tSize + 1; k <= i && k >= startY; k++) {
					stds.push_back(hsvImage.at<cv::Vec3b>(k, j)[2]);
				}
				double result = Mean(stds);
				if (abs(result - save) > 5) {

					for (int k = i; k > i - tSize * 2 && k >= startY; k--) {
						images[5].at<cv::Vec3b>(k, j)[0] = 0;
						images[5].at<cv::Vec3b>(k, j)[1] = 0;
						images[5].at<cv::Vec3b>(k, j)[2] = 0;
					}
					points.push_back({ j  , i - tSize * 2 });
					break;
				}
				else {
					images[5].at<cv::Vec3b>(i, j)[0] = 0;
					images[5].at<cv::Vec3b>(i, j)[1] = 0;
					images[5].at<cv::Vec3b>(i, j)[2] = 0;
					save = (result + save) / 2;
				}
			}
		}
		// Parcours de bas en haut
		for (int j = startX; j <= endX; j++) {
			double save = 0.0;
			std::vector<double> stds;
			for (int k = 0; k < tSize; k++) {
				stds.push_back(hsvImage.at<cv::Vec3b>(startY + k, j)[2]);
			}
			save = Mean(stds);
			for (int i = startY; i <= endY; i++) {
				stds.clear();
				for (int k = i; k < i + tSize && k <= endY; k++) {
					stds.push_back(hsvImage.at<cv::Vec3b>(k, j)[2]);
				}
				double result = Mean(stds);
				if (abs(result - save) > 5) {

					for (int k = i; k < i + tSize * 2 && k <= endY; k++) {
						images[5].at<cv::Vec3b>(k, j)[0] = 0;
						images[5].at<cv::Vec3b>(k, j)[1] = 0;
						images[5].at<cv::Vec3b>(k, j)[2] = 0;
					}
					points.push_back({ j  , i + tSize * 2 });
					break;
				}
				else {
					images[5].at<cv::Vec3b>(i, j)[0] = 0;
					images[5].at<cv::Vec3b>(i, j)[1] = 0;
					images[5].at<cv::Vec3b>(i, j)[2] = 0;
					save = (result + save) / 2;
				}
			}
		}

		cv::Mat grayscaleImage;
		cv::cvtColor(images[5], grayscaleImage, cv::COLOR_BGR2GRAY); // Convertir en niveaux de gris

		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(grayscaleImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//std::sort(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b) {
		//	if (a.x != b.x) return a.x < b.x;
		//	return a.y < b.y;
		//	});
		cv::Mat mask = cv::Mat::zeros(originalImage.size(), CV_8UC1);
		//std::vector<std::vector<cv::Point>> contours;
		//contours.push_back(points);
		cv::fillPoly(mask, contours, cv::Scalar(255));
		cv::Mat result;
		originalImage.copyTo(result, mask);

		images[5] = result.clone();


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

void ImageUtils::SaveAFromToDirectory(std::string& source, std::string& destination, int generation)
{
	// Check if source and destination are provided
	if (source.empty() || destination.empty() || !std::filesystem::exists(source) || !std::filesystem::is_directory(destination)) {
		throw std::runtime_error("Missing source or destination directory.");
	}

	std::vector<std::string> names = ImageUtils::GetImagesInDirectory(source, generation);
	for (auto i = 0; i < names.size(); i++) {
		// Load an image from the specified file path
		cv::Mat image = cv::imread(source + names[i], cv::IMREAD_COLOR);
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
		ImageProcessing::Blur(images[1], 1.5, 2.0);
		ImageProcessing::Contrast(images[2], 1.5, 2.0);
		ImageProcessing::Scale(images[3], 1.25, 1.5);
		ImageProcessing::Illumination(images[4], 30, 40);
		ImageProcessing::Projective(images[5], 30, 40);

		// Create a mosaic image from the processed images
		std::vector<std::string> augmentations = { "Rotate", "Blur", "Contrast", "Scale", "Illumination", "Projective" };
		ImageUtils::SaveImages(destination + names[i], images, augmentations);

		// Progression
		int progress = (i + 1) * 100 / names.size();
		int numComplete = (progress * 50) / 100;
		int numRemaining = 50 - numComplete;
		std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
		std::cout << "\033[A";
	}
	std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush << std::endl << "\r\033[K";
}