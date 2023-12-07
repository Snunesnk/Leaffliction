#include <iostream>
#include <filesystem>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

#include "model_utils.h"
#include "model_calculate.h"

std::vector<double> getTransformtionInfo(cv::Mat& image, const int transformation) {
	if (transformation == 1) {
		// Aspect ratio
		std::vector<cv::Point> rectanglePoints = ImageProcessing::getMinimumBoundingRectanglePoints(image);
		return { ImageProcessing::calculateAspectRatioOfObjects(image) };
	}
	else if (transformation == 2) {
		// Means colors
		std::vector<double> meansRGB(3);
		double countRGB = 0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] <= 250 || pixel[1] <= 250 || pixel[2] <= 250) {
					meansRGB[0] += pixel[0];
					meansRGB[1] += pixel[1];
					meansRGB[2] += pixel[2];
					countRGB++;
				}
			}
		}

		//// Means HSV
		//cv::Mat hsvImage;
		//cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
		//std::vector<double> meansHSV(2);
		//double countHSV = 0;
		//for (int i = 0; i < hsvImage.rows; i++) {
		//	for (int j = 0; j < hsvImage.cols; j++) {
		//		cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
		//		if (pixel[0] <= 250 || pixel[1] <= 250 || pixel[2] <= 250) {
		//			cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
		//			meansHSV[0] += hsvpixel[0];
		//			meansHSV[1] += hsvpixel[1];
		//			meansHSV[2] += hsvpixel[2];
		//			countHSV++;
		//		}
		//	}
		//}

		if (countRGB /*&& countHSV*/)
			return { 
			meansRGB[0] / countRGB, meansRGB[1] / countRGB, meansRGB[2] / countRGB,
			/*meansHSV[0] / countHSV,*/ /*meansHSV[1] / countHSV, meansHSV[2] / countHSV*/
		};
		else
			return { 0, 0, 0 };
	}
	else if (transformation <= 5) {
		// Pct selected hue 
		double mean = 0;
		double count = 0;
		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] >= 250 && pixel[1] <= 5 && pixel[2] <= 5) {
					count++;
				}
				else if (pixel[0] <= 250 || pixel[1] <= 250 || pixel[2] <= 250) {
					mean++;
					count++;
				}
			}
		}
		if (count)
			return { mean / count };
		else
			return { 0 };
	}
	else {
		// Means HSV
		cv::Mat hsvImage;
		cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);
		std::vector<double> meansHSV(2);
		double countHSV = 0;
		for (int i = 0; i < hsvImage.rows; i++) {
			for (int j = 0; j < hsvImage.cols; j++) {
				cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
				if (pixel[0] <= 250 || pixel[1] <= 250 || pixel[2] <= 250) {
					cv::Vec3b& hsvpixel = hsvImage.at<cv::Vec3b>(i, j);
					meansHSV[0] += hsvpixel[0];
					meansHSV[1] += hsvpixel[1];
					meansHSV[2] += hsvpixel[2];
					countHSV++;
				}
			}
		}
		return { meansHSV[0] / countHSV, meansHSV[1] / countHSV, meansHSV[2] / countHSV };
	}
	return {};
}

void createDatasAndCSV(std::string source, std::vector<DataInfo>& datasInfo, int generation) {
	const std::vector<std::string> features = { "T1", "T2", "T3", "T4", "T5", "T6" };
	std::vector<std::vector<std::string>> datas;

	for (const auto& entry : std::filesystem::directory_iterator(source)) {
		const std::string target = entry.path().filename().generic_string();
		// Next if not expected directory
		if (std::find(ModelUtils::types.begin(), ModelUtils::types.end(), target) == ModelUtils::types.end()) {
			continue;
		}
		const std::string source = entry.path().generic_string() + "/";
		const std::vector<std::string> files = ImageUtils::GetImagesInDirectory(source, generation);
		auto feature = -1;

		std::vector<std::string> lineElements;
		for (auto i = 0; i < files.size(); i++) {
			std::string filePath = source + files[i];
			cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
			if (image.empty()) {
				throw std::runtime_error("Unable to load the image.");
			}
			// New line
			if (feature == -1) {
				lineElements.push_back(std::to_string(datas.size()));
				lineElements.push_back(target);
				auto position = files[i].find_last_of('_');
				if (position == std::string::npos) {
					position = files[i].find_last_of('.');
				}
				lineElements.push_back(files[i].substr(0, position));
				feature++;
			}
			// Features
			for (auto j = 0; j < features.size(); j++) {
				size_t position = files[i].find(features[j]);
				if (position != std::string::npos) {
					std::vector<double> result = getTransformtionInfo(image, features[j].back() - '0');
					for (auto k = 0; k < result.size(); k++) {
						lineElements.push_back(features[j].back() + std::to_string(k) + "V" + std::to_string(result[k]));
					}
					feature++;
					break;
				}
			}
			// Line filePath
			std::cout << "\r\033[K" << filePath;
			// When line completed, push it and start new line
			if (feature == features.size()) {
				feature = -1;
				datas.push_back(lineElements);
				lineElements.clear();
				// Progression
				int progress = (i + 1) * 100 / files.size();
				int numComplete = (progress * 50) / 100;
				int numRemaining = 50 - numComplete;
				std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%" << std::flush;
				std::cout << "\033[A";
			}
		}
		std::cout << "\r\033[K[" << std::string(50, '=') << "] " << std::setw(3) << 100 << "%" << std::flush << std::endl << "\r\033[K";
	}

	// Sort features
	for (auto i = 0; i < datas.size(); i++) {
		std::vector<std::string> line = datas[i];
		line.erase(line.begin(), line.begin() + 3);
		std::sort(line.begin(), line.end());

		for (auto j = 0; j < line.size(); j++) {
			datas[i][j + 3] = line[j].erase(0, line[j].find_last_of('V') + 1);
		}
	}

	//// Print some lines
	//auto counter = 0;
	//for (const auto& row : datas) {
	//	for (const std::string& value : row) {
	//		std::cout << value << " ";
	//	}
	//	std::cout << std::endl;
	//	if (++counter == 5) break;
	//}

	for (auto i = 0; i < datas.size(); i++) {
		datasInfo.push_back(DataInfo());
		datasInfo.back().index = std::stoi(datas[i][0]);
		datasInfo.back().labels.push_back(datas[i][1]);
		datasInfo.back().labels.push_back(datas[i][2]);
		for (auto j = 0; j < datas[i].size() - 3; j++) {
			datasInfo.back().features.push_back(std::stod(datas[i][j + 3]));
		}
	}

	//// Save CSV
	//std::string filename = "data.csv";
	//std::ofstream outputfile(filename);
	//if (outputfile.is_open()) {
	//	for (size_t i = 0; i < datas.size(); i++) {
	//		for (size_t j = 0; j < datas[i].size(); j++) {
	//			outputfile << datas[i][j];
	//			if (j < datas[i].size() - 1) {
	//				outputfile << ",";
	//			}
	//			else {
	//				outputfile << std::endl;
	//			}
	//		}
	//	}
	//	outputfile.close();
	//	std::cout << "Datas saved to " << filename << std::endl;
	//}
	//else {
	//	throw std::runtime_error("Unable to open " + filename);
	//}
}

int main(int argc, char* argv[]) {
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " [-gen <generation_max>]");
		}
		std::string source = argv[1];
		int generation = 1640;

		// Parse command-line arguments
		for (int i = 2; i < argc; ++i) {
			std::string arg = argv[i];
			if (arg == "-gen" && i + 1 < argc) {
				try {
					generation = std::atoi(argv[i + 1]);
					if (generation > 1640) {
						generation = 1640;
					}
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

#ifdef _MSC_VER
		//images
		//	|--- Apple_Black_rot
		//	|    '--- 620 files
		//	|--- Apple_healthy
		//	|    '--- 1640 files
		//	|--- Apple_rust
		//	|    '--- 275 files
		//	|--- Apple_scab
		//	|    '--- 629 files
		//	|--- Grape_Black_rot
		//	|    '--- 1178 files
		//	|--- Grape_Esca
		//	|    '--- 1382 files
		//	|--- Grape_healthy
		//	|    '--- 422 files
		//	|--- Grape_spot
		//	|    '--- 1075 files
		//	'--- 0 files
		generation = 50;
#endif

		std::vector<DataInfo> datasInfo;

		bool checker = false;
		for (const auto& entry : std::filesystem::directory_iterator("./")) {
			std::filesystem::path entryPath = entry.path();
			if (std::filesystem::is_regular_file(entryPath)) {
				if (entryPath.filename().generic_string() == "data.csv") {
					checker = true;
					break;
				}
			}
		}
		if (checker == false) {

			// Augmentations limited by -gen
			std::cout << "Doing augmentations if needed..." << std::endl;
			int augmentationCout = 0;
			for (const auto& entry : std::filesystem::directory_iterator(source)) {
				std::filesystem::path entryPath = entry.path();
				if (std::filesystem::is_directory(entryPath)) {

					const std::string target = entry.path().filename().generic_string();				
					// Next if not expected directory
					if (std::find(ModelUtils::types.begin(), ModelUtils::types.end(), target) == ModelUtils::types.end()) {
						continue;
					}

					std::string directoryPath = entryPath.generic_string() + "/";
					for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
						if (std::filesystem::is_regular_file(entry)) {
							std::string filename = entry.path().filename().string();
							if (filename.find('_') != std::string::npos && entry.is_regular_file() && entry.path().extension() == ".JPG") {
								std::filesystem::remove(entry.path());
								std::cout << "Delete file : " << entry.path() << "\r\033[K";
							}
						}
					}
					int tmp_generation = generation;
					for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
						if (entry.is_regular_file() && entry.path().extension() == ".JPG") {
							if (--tmp_generation == 0) {
								break;
							}
						}
					}
					
					if (tmp_generation > 0) {
						augmentationCout += tmp_generation;
						ImageUtils::SaveAFromToDirectory(directoryPath, directoryPath, tmp_generation);
					}
				}
			}
			std::cout << "Augmentations generated : " << augmentationCout << std::endl;
			// Transformations limited by -gen
			std::cout << "Doing transformation..." << std::endl;
			for (const auto& entry : std::filesystem::directory_iterator(source)) {
				std::filesystem::path entryPath = entry.path();
				if (std::filesystem::is_directory(entryPath)) {
					std::string directoryPath = entryPath.generic_string() + "/";

					ImageUtils::SaveTFromToDirectory(directoryPath, directoryPath, generation);
				}
			}
			std::cout << "Transformations generated : " << generation * 8 * 6 << std::endl;
			std::cout << "Doing data generation..." << std::endl;
			createDatasAndCSV(source, datasInfo, generation * 7);
		}
		else {
			std::cout << "Loading data.csv..." << std::endl;
			ModelUtils::LoadDataFile("data.csv", datasInfo);
		}
		std::cout << "Creating model..." << std::endl;
		ModelCalculate::CreateModel(datasInfo);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}