#include <iostream>
#include <filesystem>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

#include "model_utils.h"
#include "model_calculate.h"

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

		if (directoryPath.back() != '/') {
			directoryPath += "/";
		}

		std::vector<std::string> labels = { "T1", "T2", "T3", "T4", "T5", "T6" };
		std::vector<double(*)(cv::Mat& image)> functions = { getAspectRatio, getRedMean, getGreenMean, getBlueMean, getSaturationPtc, getValuePtc };
		const int featureBegin = 3;
		std::vector<std::vector<std::string>> datas;
		auto prevLine = 0;
		for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
			break;
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
			std::vector<std::string> elems(labels.size() + featureBegin, "");
			for (auto i = 0; i < files.size(); i++) {
				std::string filePath = entryPath + files[i];
				cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
				if (image.empty()) {
					std::cerr << "Unable to load the image. " << filePath << std::endl;
					return 1;
				}

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

				for (auto j = 0; j < labels.size(); j++) {
					size_t position = files[i].find(labels[j]);
					if (position != std::string::npos) {
						double result = functions[j](image);
						elems[j + featureBegin] = std::to_string(result);
						feature++;
					}
				}

				if (feature == labels.size()) {
					feature = -1;
					datas.push_back(elems);
					//elems = std::vector<std::string>(labels.size() + featureBegin, "");
				}
			}
			if (feature != -1) {
				std::cerr << "Unable to create data: transformation missing. " << std::endl;
				return 1;
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

		std::vector<DataInfo> datasInfo;
		auto [headers, featuresStartIndex] = ModelUtils::LoadDataFile("data.csv", datasInfo);
		
		for (auto i = 0; i < datas.size(); i++) {
			datasInfo.push_back(DataInfo());
			datasInfo.back().index = std::stoi(datas[i][0]);
			datasInfo.back().labels.push_back(datas[i][1]);
			for (auto j = 0; j < labels.size(); j++) {
				datasInfo.back().features.push_back(std::stod(datas[i][1]));
			}
		}
		ModelCalculate::CreateModel(datasInfo);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}