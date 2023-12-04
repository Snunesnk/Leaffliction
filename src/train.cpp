#include <iostream>
#include <filesystem>
#include "image_processing.h"
#include "image_utils.h"

double getAspectRatio(cv::Mat& image) {
	std::vector<cv::Point> rectanglePoints = ImageProcessing::getMinimumBoundingRectanglePoints(image);
	// Dessinez le parallélogramme sur l'image
	for (int i = 0; i < rectanglePoints.size(); ++i) {
		cv::line(image, rectanglePoints[i], rectanglePoints[(i + 1) % rectanglePoints.size()], cv::Scalar(255, 0, 255), 2);
	}
	return ImageProcessing::calculateAspectRatioOfObjects(image);
}

double getRedMean(cv::Mat& image) {
	double mean = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				mean += pixel[2];
				count++;
			}
		}
	}
	return mean / count;
}
double getGreenMean(cv::Mat& image) {
	double mean = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				mean += pixel[1];
				count++;
			}
		}
	}
	return mean / count;
}
double getBlueMean(cv::Mat& image) {
	double mean = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				mean += pixel[0];
				count++;
			}
		}
	}
	return mean / count;
}

double getSaturationPtc(cv::Mat& image) {
	double valid = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				count++;
				if (pixel[0] + pixel[1] + pixel[2]) {
					valid++;
				}
			}
		}
	}
	return count / valid;
}

double getValuePtc(cv::Mat& image) {
	double valid = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
			if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
				count++;
				if (pixel[0] + pixel[1] + pixel[2]) {
					valid++;
				}
			}
		}
	}
	return count / valid;
}

int main(int argc, char* argv[]) {
	try {
		if (argc != 2) {
			std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
			return 1;
		}
		std::string directoryPath = argv[1];
		std::cout << directoryPath << std::endl;
#ifdef _MSC_VER
		directoryPath = "images/test2";
#endif
		if (directoryPath.back() != '/') {
			directoryPath += "/";
		}

		std::vector<std::string> labels = { "Shape", "RedChannel", "GreenChannel", "BlueChannel", "Saturation", "Value" };
		std::vector<double(*)(cv::Mat& image)> functions = { getAspectRatio, getRedMean, getGreenMean, getBlueMean, getSaturationPtc, getValuePtc };

		std::vector<std::string> files = ImageUtils::GgetImagesInDirectory(directoryPath);
		for (auto file : files) {
			std::string filePath = directoryPath + file;
			cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
			if (image.empty()) {
				std::cerr << "Unable to load the image. " << filePath << std::endl;
				break;
			}
			std::cout << "Image opened. " << filePath << std::endl;

			// Utilisez la méthode find pour chercher le mot dans le texte
			for (auto i = 0; i < labels.size(); i++) {
				size_t position = filePath.find(labels[i]);
				if (position != std::string::npos) {				
					double aspectRatio = functions[i](image);
					cv::imshow("image", image);
					std::cout << labels[i] << " : " << aspectRatio << std::endl;
					cv::waitKey(0);
					break;
				}
			}
		}
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}