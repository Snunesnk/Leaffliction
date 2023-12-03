#include <iostream>
#include <filesystem>
#include "image_processing.h"
#include "image_utils.h"

int main(int argc, char* argv[]) {
	try {
		if (argc != 2) {
			std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
			return 1;
		}

		std::string directoryPath = argv[1];
		std::cout << directoryPath << std::endl;

		size_t value = std::stoi(output_pair[i].second);
		for (auto j = 0; j < value; j++) {
			std::string filePath = argv[1];
			std::string index = std::to_string(j + 1);
			filePath += "/" + output_pair[i].first + "/image (" + index + ").JPG";
			cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
			if (image.empty())
			{
				std::cerr << "Unable to load the image. " << filePath << std::endl;

				continue;
			}
			else
				std::cout << "Image opened. " << filePath << std::endl;


			std::vector<cv::Mat> images;

			cv::Mat tmp_image = image.clone();
			cv::Scalar lower_leaf_colors(10, 40, 20);
			cv::Scalar upper_leaf_colors(100, 255, 255);
			//ImageProcessing::applyColorFiltering(tmp_image, lower_leaf_colors, upper_leaf_colors);
			//std::vector<cv::Point> convexPoints = ImageProcessing::getConvexHullPoints(tmp_image);
			//ImageProcessing::cropImageWithPoints(image, convexPoints);
			//std::vector<cv::Point> rectanglePoints = ImageProcessing::getMinimumBoundingRectanglePoints(tmp_image);
			//ImageProcessing::drawAndScaleRectangleFromPoints(images[1], rectanglePoints);
			//std::cout << ImageProcessing::calculateAspectRatioOfObjects(tmp_image) << std::endl;

			cv::waitKey(0);

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