#include <iostream>
#include <filesystem>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

#include "model_utils.h"
#include "model_calculate.h"

std::vector<double> getTransformtionInfo(cv::Mat& image, const int transformation) {
	if (transformation == 1) { //Shape
		std::vector<cv::Point> rectanglePoints = ImageProcessing::getMinimumBoundingRectanglePoints(image);
		// Dessinez le parallélogramme sur l'image
		for (int i = 0; i < rectanglePoints.size(); ++i) {
			cv::line(image, rectanglePoints[i], rectanglePoints[(i + 1) % rectanglePoints.size()], cv::Scalar(255, 0, 255), 2);
		}
		return { ImageProcessing::calculateAspectRatioOfObjects(image) };
	}
	else if (transformation == 2) //HEC
	{
		std::vector<double> means(3);
		double count = 0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
					means[0] += pixel[0];
					means[1] += pixel[1];
					means[2] += pixel[2];
					count++;
				}
			}
		}
		return { means[0] / count, means[1] / count, means[2] / count };
	}
	else if (transformation == 3) { //HES
		cv::Mat hsvImage;
		cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
		double mean = 0;
		int count = 0;
		for (int i = 0; i < hsvImage.rows; i++) {
			for (int j = 0; j < hsvImage.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
					cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
					mean += hsvpixel[1];
					count++;
				}
			}
		}
		return { mean / count };
	}
	else if (transformation == 4) { //HEV
		cv::Mat hsvImage;
		cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
		double mean = 0;
		int count = 0;
		for (int i = 0; i < hsvImage.rows; i++) {
			for (int j = 0; j < hsvImage.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
					cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
					mean += hsvpixel[2];
					count++;
				}
			}
		}
		return { mean / count };
	}
	else if (transformation == 5) { //Grey
		double mean = 0;
		double count = 0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
					mean += pixel[0];
					count++;
				}
			}
		}
		return { mean / count };
	}
	else if (transformation == 6) { //HEG
		double mean = 0;
		double count = 0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] + pixel[1] + pixel[2] - 255 * 3) {
					mean += pixel[0];
					count++;
				}
			}
		}
		return { mean / count };
	}
	return {};
}

void createDatasAndCSV(std::string source, std::vector<DataInfo>& datasInfo, int generation) {
	std::vector<std::vector<std::string>> datas;
	std::vector<std::string> labels = { "T1", "T2", "T3", "T4", "T5", "T6" };
	std::vector<int> positions = { 3, 4, 7, 8, 9, 10 };
	for (const auto& entry : std::filesystem::directory_iterator(source)) {

		const std::string target = entry.path().filename().generic_string();
		bool v = false;
		for (auto j = 0; j < ModelUtils::targets.size(); j++) {
			if (target == ModelUtils::targets[j]) {
				v = true;
				break;
			}
		}
		if (v == false) {
			continue;
		}
		const std::string entryPath = entry.path().generic_string() + "/";
		const std::vector<std::string> files = ImageUtils::GgetImagesInDirectory(entryPath);

		auto feature = -1;
		std::vector<std::string> elems(positions.back() + 1, "");
		for (auto i = 0; i < files.size(); i++) {
			std::string filePath = entryPath + files[i];
			cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
			if (image.empty()) {
				throw std::runtime_error("Unable to load the image.");
			}
			bool checker = false;
			// New line
			if (feature == -1) {
				elems[0] = std::to_string(datas.size());
				elems[1] = target;
				auto position = files[i].find_last_of('_');
				if (position == std::string::npos) {
					position = files[i].find_last_of('.');
				}
				const std::string name = files[i].substr(0, position);
				elems[2] = name;
				feature++;
			}
			// Features
			for (auto j = 0; j < labels.size(); j++) {
				size_t position = files[i].find(labels[j]);
				if (position != std::string::npos) {
					std::vector<double> result = getTransformtionInfo(image, j);
					for (auto k = 0; k < result.size(); k++) {
						elems[positions[j] + k] = std::to_string(result[k]);
					}
					feature++;
					checker = true;
				}
			}
			// When line completed, push it and start new line
			if (feature == labels.size()) {
				feature = -1;
				datas.push_back(elems);
				if (--generation == 0) {
					break;
				}
			}
			else if (checker == false) {
				throw std::runtime_error("Unable to create data: image missing.");
			}
		}
	}

	//// Nom du fichier CSV
	//std::string filename = "data.csv";

	//// Ouvrir le fichier en mode écriture
	//std::ofstream outputFile(filename);

	//if (outputFile.is_open()) {
	//	// Écriture des données dans le fichier CSV
	//	for (size_t i = 0; i < datas.size(); i++) {
	//		for (size_t j = 0; j < datas[i].size(); j++) {
	//			outputFile << datas[i][j];
	//			if (j < datas[i].size() - 1) {
	//				outputFile << ",";
	//			}
	//			else {
	//				outputFile << std::endl;
	//			}
	//		}
	//	}

	//	// Fermer le fichier
	//	outputFile.close();

	//	std::cout << "Datas saved to " << filename << std::endl;
	//}
	//else {
	//	std::cerr << "Unable to open " << filename << std::endl;
	//}

	auto [headers, featuresStartIndex] = ModelUtils::LoadDataFile("data.csv", datasInfo);

	for (auto i = 0; i < datas.size(); i++) {
		datasInfo.push_back(DataInfo());
		datasInfo.back().index = std::stoi(datas[i][0]);
		datasInfo.back().labels.push_back(datas[i][1]);
		for (auto j = 0; j < labels.size(); j++) {
			datasInfo.back().features.push_back(std::stod(datas[i][1]));
		}
	}
}

int main(int argc, char* argv[]) {
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " [-gen <generation_max>]");
		}
		std::string source = argv[1];
		int generation = std::numeric_limits<int>::max();

		// Parse command-line arguments
		for (int i = 2; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-gen" && i + 1 < argc) {
				try {
					generation = std::atoi(argv[i + 1]);
				}
				catch (...) {
					throw std::runtime_error("Unable to read the -gen value.");
				}
				++i;
			}
			else if (arg == "-h") {
				std::cout << "Usage: " << argv[0] << " [-gen <generation_max>]" << std::endl;
				return 0;
			}
		}

		if (source.back() != '/') {
			source += "/";
		}
		std::cout << source << std::endl;
		std::vector<DataInfo> datasInfo;
		// Augmentations based on generation -> function needed
		// Transformations based on generation -> function needed
		createDatasAndCSV(source, datasInfo, generation);
		ModelCalculate::CreateModel(datasInfo);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}