#include <iostream>
#include <filesystem>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

#include "model_utils.h"
#include "model_calculate.h"

std::vector<double> getTransformtionInfo(cv::Mat& image, const int transformation)
{
	if (transformation == 1) {
		// Aspect ratio
		std::vector<cv::Point> rectanglePoints = ImageProcessing::getMinimumBoundingRectanglePoints(image);
		return { ImageProcessing::calculateAspectRatioOfObjects(image) };
	}
	else if (transformation == 2) {
		// Means colors
		std::vector<std::vector<double>> RGBValues(3);
		cv::Mat HSVImage;
		cv::cvtColor(image, HSVImage, cv::COLOR_BGR2HSV);
		std::vector<std::vector<double>> HSVValues(3);

		for (int i = 0; i < image.rows; i++) {
			for (int j = 0; j < image.cols; j++) {
				cv::Vec3b& RGBpixel = image.at<cv::Vec3b>(i, j);
				if (RGBpixel[0] <= 250 || RGBpixel[1] <= 250 || RGBpixel[2] <= 250) {
					RGBValues[0].push_back(RGBpixel[0]);
					RGBValues[1].push_back(RGBpixel[1]);
					RGBValues[2].push_back(RGBpixel[2]);
					cv::Vec3b& HSVpixel = HSVImage.at<cv::Vec3b>(i, j);
					HSVValues[0].push_back(HSVpixel[0]);
					HSVValues[1].push_back(HSVpixel[1]);
					HSVValues[2].push_back(HSVpixel[2]);
				}
			}
		}
		std::vector<double> RGBMeans(3);
		RGBMeans[0] = ImageUtils::Mean(RGBValues[0]);
		RGBMeans[1] = ImageUtils::Mean(RGBValues[1]);
		RGBMeans[2] = ImageUtils::Mean(RGBValues[2]);

		std::vector<double> RGBStandardDeviation(3);
		RGBStandardDeviation[0] = ImageUtils::Mean(RGBValues[0]);
		RGBStandardDeviation[1] = ImageUtils::Mean(RGBValues[1]);
		RGBStandardDeviation[2] = ImageUtils::Mean(RGBValues[2]);

		double RGMeansDiff = RGBMeans[1] - RGBMeans[2];

		std::vector<double> HSVMeans(3);
		HSVMeans[0] = ImageUtils::Mean(HSVValues[0]);
		HSVMeans[1] = ImageUtils::Mean(HSVValues[1]);
		HSVMeans[2] = ImageUtils::Mean(HSVValues[2]);

		std::vector<double> numerics = {
			RGBMeans[0], RGBMeans[1] ,RGBMeans[2], /*RGMeansDiff,
			RGBStandardDeviation[0], RGBStandardDeviation[1], RGBStandardDeviation[2],
			HSVMeans[0], HSVMeans[1], HSVMeans[2],*/
		};


		// La moyenne des valeurs de pixels des canaux (rouge, vert, bleu) pour chaque bande de fréquence de Fourier 
		cv::Mat bgrChannels[3];
		cv::split(image, bgrChannels);

		// Calculer la 2D DFT pour chaque canal
		for (int channel = 0; channel < 3; channel++) {
			cv::Mat grayChannel;
			bgrChannels[channel].convertTo(grayChannel, CV_32F); // Conversion en float
			cv::Mat complexChannel;
			cv::dft(grayChannel, complexChannel, cv::DFT_COMPLEX_OUTPUT);

			// Séparer les parties réelles et imaginaires
			std::vector<cv::Mat> channels(2);
			cv::split(complexChannel, channels);
			cv::Mat realPart = channels[0];

			// Calculer la moyenne des valeurs de la partie réelle pour chaque bande de fréquence
			int numBands = complexChannel.cols;
			std::vector<double> DFTMeans(numBands, 0.0);

			for (int i = 0; i < numBands; i += 25) {
				cv::Mat band = realPart.col(i);
				cv::Scalar mean = cv::mean(band);
				DFTMeans[i] = mean[0];
				//numerics.push_back(DFTMeans[i]);
			}
		}

		return numerics;
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
		//return { meansHSV[0] / countHSV, meansHSV[1] / countHSV, meansHSV[2] / countHSV };
	}
	return { 0 };
}

void CreateData(std::string source, std::vector<DataInfo>& datasInfo, int generation)
{
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
			// Features // REFAIRE L ORDRE DES IMAGES CAR PROB AVEC AUGS
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
				int progress = (i + 1) * 100 / static_cast<int>(files.size());
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
		std::sort(line.begin(), line.end(), [](const std::string& a, const std::string& b) {
			std::string numeroA = a.substr(0, a.find('V'));
			std::string numeroB = b.substr(0, b.find('V'));
			try {
				int intA = std::stoi(numeroA);
				int intB = std::stoi(numeroB);
				return intA < intB;
			}
			catch (...) {
				throw std::runtime_error("Error: sort features");
			}
			}
		);

		for (auto j = 0; j < line.size(); j++) {
			datas[i][j + 3] = line[j].erase(0, line[j].find_last_of('V') + 1);
		}
	}

	for (auto i = 0; i < datas.size(); i++) {
		datasInfo.push_back(DataInfo());
		datasInfo.back().index = std::stoi(datas[i][0]);
		datasInfo.back().labels.push_back(datas[i][1]);
		datasInfo.back().labels.push_back(datas[i][2]);

		for (auto j = 0; j < datas[i].size() - 3; j++) {
			datasInfo.back().features.push_back(std::stod(datas[i][j + 3]));
		}
	}
	std::cout << std::endl;
	ModelUtils::SaveDataFile("data.csv", datas);
}

int main(int argc, char* argv[])
{
	try {
		if (argc < 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " [-gen <generation_max>]");
		}
		std::string source = argv[1];
		int generation = 100;

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
		generation = 200;

#endif
		auto step = 0; // 0: create images and data. 1: only create data.
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
			if (step <= 0) {
				// Delete existing generated images
				std::cout << "Reseting..." << std::endl;
				int imageCout = 0;
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
									imageCout++;
								}
							}
						}
					}
				}
				std::cout << "Images deleted : " << imageCout << std::endl;

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
							ImageUtils::SaveAFromToDirectory(directoryPath, directoryPath, tmp_generation / 6);
						}
					}
				}
				std::cout << "Augmentations generated : " << augmentationCout << std::endl;

				// Transformations limited by -gen
				std::cout << "Doing transformation..." << std::endl;
				for (const auto& entry : std::filesystem::directory_iterator(source)) {
					const std::string target = entry.path().filename().generic_string();
					// Next if not expected directory
					if (std::find(ModelUtils::types.begin(), ModelUtils::types.end(), target) == ModelUtils::types.end()) {
						continue;
					}
					std::filesystem::path entryPath = entry.path();
					if (std::filesystem::is_directory(entryPath)) {
						std::string directoryPath = entryPath.generic_string() + "/";
						ImageUtils::SaveTFromToDirectory(directoryPath, directoryPath, generation);
					}
				}
				std::cout << "Transformations generated : " << generation * 8 * 6 << std::endl;
			}
			if (step <= 1) {
				std::cout << "Doing data generation..." << std::endl;
				CreateData(source, datasInfo, generation * 7);
			}
		}
		else {
			std::cout << "Loading data.csv..." << std::endl;
			ModelUtils::LoadDataFile(datasInfo, "data.csv");
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