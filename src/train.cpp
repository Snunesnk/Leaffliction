#include <iostream>
#include <filesystem>
#include <fstream>
#include "image_processing.h"
#include "image_utils.h"

#include "model_utils.h"
#include "model_calculate.h"

#define M_PI 3.14159265358979323846264338327950288419716939937510582

// Fonction personnalisée pour calculer la médiane d'une image
double medianMat(const cv::Mat& src)
{
	std::vector<uchar> array;
	if (src.isContinuous()) {
		array.assign(src.datastart, src.dataend);
	}
	else {
		for (int i = 0; i < src.rows; ++i) {
			array.insert(array.end(), src.ptr<uchar>(i), src.ptr<uchar>(i) + src.cols);
		}
	}
	std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
	return array[array.size() / 2];
}

// Function to calculate the Gray Level Co-occurrence Matrix (GLCM)
cv::Mat calculateGLCM(const cv::Mat& img, int dx, int dy, int numLevels)
{
	cv::Mat glcm = cv::Mat::zeros(numLevels, numLevels, CV_32F);

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			int rowValue = img.at<uchar>(y, x);
			int colValue = 0;

			if ((y + dy) >= 0 && (y + dy) < img.rows && (x + dx) >= 0 && (x + dx) < img.cols) {
				colValue = img.at<uchar>(y + dy, x + dx);
			}

			// Check if rowValue and colValue are within the bounds of the GLCM matrix
			if (rowValue >= 0 && rowValue < numLevels && colValue >= 0 && colValue < numLevels) {
				glcm.at<float>(rowValue, colValue) += 1.0f;
			}
		}
	}

	glcm = glcm / cv::sum(glcm)[0];
	return glcm;
}



// Fonction pour extraire des caractéristiques à partir de la GLCM
std::vector<double> extractGLCMFeatures(const cv::Mat& glcm)
{
	double contrast = 0.0;
	double dissimilarity = 0.0;
	double homogeneity = 0.0;
	double energy = 0.0;
	double correlation = 0.0;
	double mean_i = 0.0;
	double mean_j = 0.0;
	double std_i = 0.0;
	double std_j = 0.0;

	// Calculer les moyennes et les écarts-types
	for (int i = 0; i < glcm.rows; i++) {
		for (int j = 0; j < glcm.cols; j++) {
			mean_i += i * glcm.at<float>(i, j);
			mean_j += j * glcm.at<float>(i, j);
		}
	}

	for (int i = 0; i < glcm.rows; i++) {
		for (int j = 0; j < glcm.cols; j++) {
			std_i += glcm.at<float>(i, j) * (i - mean_i) * (i - mean_i);
			std_j += glcm.at<float>(i, j) * (j - mean_j) * (j - mean_j);
		}
	}

	std_i = sqrt(std_i);
	std_j = sqrt(std_j);

	// Calculer les caractéristiques
	for (int i = 0; i < glcm.rows; i++) {
		for (int j = 0; j < glcm.cols; j++) {
			double p = glcm.at<float>(i, j);
			contrast += p * (i - j) * (i - j);
			dissimilarity += p * abs(i - j);
			homogeneity += p / (1.0 + abs(i - j));
			energy += p * p;
			if (std_i != 0.0 && std_j != 0.0) {
				correlation += (i * j * p - mean_i * mean_j) / (std_i * std_j);
			}
		}
	}

	return { contrast, dissimilarity, homogeneity, energy, correlation };
}

// Fonction pour extraire toutes les caractéristiques d'une image en niveaux de gris
std::vector<double> extractFeaturesFromImageT1(const cv::Mat& image)
{
	std::vector<double> features;

	cv::Mat grayImage;
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	//cv::Mat grayscaleImage = grayImage.clone();
	//grayscaleImage.convertTo(grayscaleImage, CV_8U);
	//cv::threshold(grayscaleImage, grayscaleImage, 1, 255, cv::THRESH_BINARY);
	//double area = cv::countNonZero(grayscaleImage);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(grayImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	double maxPerimeter = 0.0;
	for (const auto& contour : contours) {
		double perimeter = cv::arcLength(contour, true);
		if (perimeter > maxPerimeter) {
			maxPerimeter = perimeter;
		}
	}

	//features.push_back(area);
	features.push_back(maxPerimeter);
	features.push_back(ImageProcessing::calculateAspectRatioOfObjects(image));

	//// Nombre de composantes connectées
	//cv::Mat labeledImage;
	//int numConnectedComponents = cv::connectedComponents(grayImage, labeledImage, 8, CV_32S);
	//cv::Moments moments = cv::moments(contours[0]);
	//double huMoments[7];
	//cv::HuMoments(moments, huMoments);
	//features.push_back(numConnectedComponents);
	//for (int i = 0; i < 7; ++i) {
	//	features.push_back(huMoments[i]);
	//}

	// Statistiques de luminance
	cv::Scalar mean, stddev;
	cv::meanStdDev(image, mean, stddev);
	double meanLuminance = mean.val[0];
	double medianLuminance = medianMat(image);
	double varianceLuminance = stddev.val[0] * stddev.val[0];
	double stdDevLuminance = stddev.val[0];
	features.push_back(meanLuminance);
	features.push_back(medianLuminance);
	features.push_back(varianceLuminance);
	features.push_back(stdDevLuminance);

	//// Histogramme des niveaux de luminance
	//std::vector<int> histogram(256, 0);
	//for (int y = 0; y < image.rows; ++y) {
	//	for (int x = 0; x < image.cols; ++x) {
	//		int pixelValue = static_cast<int>(image.at<uchar>(y, x));
	//		histogram[pixelValue]++;
	//	}
	//}

	//for (int i = 0; i < 256; i += 25) {
	//	features.push_back(static_cast<double>(histogram[i]));
	//}

	//// Caractéristiques de texture (GLCM)
	//cv::Mat gray;
	//cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	//cv::Mat yChannelNormalized;
	//cv::normalize(gray, yChannelNormalized, 0, 255, cv::NORM_MINMAX, CV_8U);
	//int numLevels = 8; // Ajustez selon vos besoins
	//cv::Mat glcm = calculateGLCM(yChannelNormalized, 1, 1, numLevels); // Calculer la GLCM
	//std::vector<double> glcmFeatures = extractGLCMFeatures(glcm); // Extraire des caractéristiques de la GLCM
	//features.insert(features.end(), glcmFeatures.begin(), glcmFeatures.end()); // Ajouter les caractéristiques de GLCM aux caractéristiques

	return features;
}

// Fonction pour extraire toutes les caractéristiques d'une image binaire
std::vector<double> extractFeaturesFromImageT2(const cv::Mat& binaryImage)
{
	std::vector<double> features;

	return features;
}


// Fonction pour extraire toutes les caractéristiques d'une image du canal Y
std::vector<double> extractFeaturesFromImageT3(const cv::Mat& yChannelImage)
{
	std::vector<double> features;

	return features;
}

// Fonction pour extraire toutes les caractéristiques d'une image de contours
std::vector<double> extractFeaturesFromImageT4(const cv::Mat& contourImage)
{
	std::vector<double> features;

	return features;
}

// Fonction pour extraire toutes les caractéristiques d'une image après amélioration du contraste
std::vector<double> extractFeaturesFromImageT5(const cv::Mat& enhancedImage)
{
	std::vector<double> features;

	return features;
}

// Fonction pour extraire toutes les caractéristiques d'une image après flou gaussien
std::vector<double> extractFeaturesFromImageT6(const cv::Mat& blurredImage)
{
	std::vector<double> features;

	return features;
}

std::vector<double> getTransformtionInfo(cv::Mat& image, const int transformation)
{

	if (transformation == 1) {
		return extractFeaturesFromImageT1(image);
	}
	else if (transformation == 2) {
		//return extractFeaturesFromImageT2(image);
	}
	else if (transformation == 3) {
		//return extractFeaturesFromImageT3(image);
	}
	else if (transformation == 4) {
		//return extractFeaturesFromImageT4(image);		
	}
	else if (transformation == 5) {
		//return extractFeaturesFromImageT5(image);
	}
	else if (transformation == 6) {	
		//return extractFeaturesFromImageT6(image);
	}
	return {  };
}

void CreateData(std::string source, std::vector<DataInfo>& dataBase, int generation)
{
	const std::vector<std::string> features = { "T1", "T2", "T3", "T4", "T5", "T6" };
	std::vector<std::vector<std::string>> dataLines;

	for (const auto& entry : std::filesystem::directory_iterator(source)) {
		const std::string targetClass = entry.path().filename().generic_string();
		// Next if not expected directory
		if (std::find(ModelUtils::types.begin(), ModelUtils::types.end(), targetClass) == ModelUtils::types.end()) {
			continue;
		}
		const std::string source = entry.path().generic_string() + "/";

		//create packs
		std::vector<std::string> files = ImageUtils::GetImagesInDirectory(source, generation);
		std::unordered_map<std::string, size_t> packIndices;
		std::vector<std::vector<std::string>> transmoPacks;
		//split
		for (auto& file : files) {
			const size_t pos = file.find_last_of('_');
			if (pos == std::string::npos || file[pos + 1] != 'T') {
				continue;
			}
			const std::string name = file.substr(0, pos);
			if (packIndices.find(name) == packIndices.end()) {
				packIndices[name] = transmoPacks.size();
				transmoPacks.push_back({ file });
			}
			else {
				transmoPacks[packIndices[name]].push_back(file);
			}
		}
		//sort
		for (auto& pack : transmoPacks) {
			std::sort(pack.begin(), pack.end(), [](const std::string& a, const std::string& b) {
				std::string numeroA = a.substr(a.length() - 5, 1);
				std::string numeroB = b.substr(b.length() - 5, 1);
				try {
					int intA = std::stoi(numeroA);
					int intB = std::stoi(numeroB);
					return intA < intB;
				}
				catch (...) {
					throw std::runtime_error("Error: stoi crash");
				}
				}
			);
		}
		files.clear();

		for (auto& pack : transmoPacks) {
			if (pack.size() != 6) {
				continue;
			}
			std::vector<std::string> dataLine;
			for (auto i = 0; i < pack.size(); i++) {
				std::string filePath = source + pack[i];
				cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);
				if (image.empty()) {
					throw std::runtime_error("Unable to load the image.");
				}
				// add new line
				if (i == 0) {
					// add index
					dataLine.push_back(std::to_string(dataLines.size()));
					// add class
					dataLine.push_back(targetClass);
					auto position = pack[i].find_last_of('_');
					// add image name
					dataLine.push_back(pack[i].substr(0, position));
				}
				// add features 
				const std::vector<double> result = getTransformtionInfo(image, i + 1);
				for (auto k = 0; k < result.size(); k++) {
					dataLine.push_back(std::to_string(result[k]));
				}
				// Display
				std::cout << "\r\033[K" << filePath;
			}
			// When line completed, push it and start new line
			dataLines.push_back(dataLine);
			// Display
			int progress = ((dataLines.size() + 1) * 100) / ((generation / 7) * 8);
			int numComplete = (progress * 50) / 100;
			int numRemaining = 50 - numComplete;
			std::cout << "\n[" << std::string(numComplete, '=') << std::string(numRemaining, ' ') << "] " << std::setw(3) << progress << "%";
			std::cout << "\033[A";
		}
		std::cout << "\033[A\r\033[K";
		std::cout << "\033[A\r\033[K";
	}
	std::cout << "\n\n\r\033[K\n\r\033[K\033[A\033[A\033[A\033[A";
	//
	for (auto i = 0; i < dataLines.size(); i++) {
		dataBase.push_back(DataInfo());
		dataBase.back().index = std::stoi(dataLines[i][0]);
		dataBase.back().labels.push_back(dataLines[i][1]);
		dataBase.back().labels.push_back(dataLines[i][2]);

		for (auto j = 0; j < dataLines[i].size() - 3; j++) {
			dataBase.back().features.push_back(std::stod(dataLines[i][j + 3]));
		}
	}
	std::cout << std::endl;
	//ModelUtils::SaveDataFile("data.csv", dataLines);
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
		// Apple_Black_rot     620 files
		// Apple_healthy       1640 files
		// Apple_rust          275 files
		// Apple_scab          629 files
		// Grape_Black_rot     1178 files
		// Grape_Esca          1382 files
		// Grape_healthy       422 files
		// Grape_spot          1075 files
		generation = 150; ///focus sur le nombre de coin

#endif
		auto step = 0; // 0: create images and data. 1: only create data.
		std::vector<DataInfo> dataBase;

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
				CreateData(source, dataBase, generation * 7);
			}
		}
		else {
			std::cout << "Loading data.csv..." << std::endl;
			ModelUtils::LoadDataFile(dataBase, "data.csv");
		}
		std::cout << "Creating model..." << std::endl;
		ModelCalculate::CreateModel(dataBase);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}