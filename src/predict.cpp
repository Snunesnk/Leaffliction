#include "image_utils.h"
#include "image_processing.h"
#include "model_utils.h"
#include "model_calculate.h"

#include <iostream>

void processImagesInDirectory(const std::string& source, const std::vector<double>& featureMeans, const std::vector<double>& featureStdDevs, const std::vector<std::vector<double>>& weights)
{
	std::vector<cv::Mat> images;

	ImageUtils::numComplete = 0;
	ImageUtils::progress = 0;
	cv::parallel_for_(cv::Range(0, ModelUtils::targets.size()), [&](const cv::Range& range) {
		for (int directory = range.start; directory < range.end; directory++) {
			std::string directoryPath = source + ModelUtils::targets[directory] + "/";
			// Get images list
			std::vector<std::string> names = ImageUtils::GetImagesInDirectory(directoryPath, 1640 * 7);
			// Check all images from list
			for (auto i = 0; i < names.size(); i++) {
				// Next if transformed image
				if (names[i].find("_T") != std::string::npos) {
					continue;
				}
				// Get image
				cv::Mat originalImage = cv::imread(directoryPath + names[i]);
				if (originalImage.empty()) {
					continue;
				}
				// Get tranformed images
				std::vector<cv::Mat> images;
				images.push_back(originalImage.clone());
				std::string name = directoryPath + names[i].substr(0, names[i].length() - 4);
				for (int i = 1; i <= 6; ++i) {
					cv::Mat image = cv::imread(name + "_T" + std::to_string(i) + ".JPG");
					if (!image.empty()) {
						images.push_back(image);
					}
					else {
						throw std::runtime_error("Unable to load transformed image.");
					}
				}
				// Create entry
				DataEntry dataEntry;
				// Add one-hot
				dataEntry.target = ModelUtils::targets[directory];
				size_t numTargets = 8;
				std::vector<double> targetOneHot(numTargets, 0);
				for (size_t target = 0; target < numTargets; ++target) {
					if (ModelUtils::targets[target] == dataEntry.target) {
						targetOneHot[target] = 1;
						break;
					}
				}
				// Add features 
				for (int i = 0; i < 7; i++) {
					std::vector<double> features;
					std::vector<double> color = ImageProcessing::ExtractColorCaracteristics(images[i]);
					std::vector<double> texture = ImageProcessing::ExtractTextureCaracteristics(images[i]);
					features.insert(features.end(), color.begin(), color.end());
					features.insert(features.end(), texture.begin(), texture.end());

					for (const auto feature : features) {
						dataEntry.features.push_back(feature);
					}
				}
				// Normalization z-score / Filter out standard deviation zero
				size_t numFeatures = dataEntry.features.size();
				std::vector<double> newFeatures;
				for (size_t i = 0; i < numFeatures; ++i) {
					if (featureStdDevs[i] != 0.0) {
						newFeatures.push_back((dataEntry.features[i] - featureMeans[i]) / featureStdDevs[i]);
					}
				}
				dataEntry.features = newFeatures;
				// Check accuracy
				double maxProbability = -1.0;
				size_t predictedTarget = 0;
				for (size_t j = 0; j < numTargets; ++j) {
					double probability = ModelCalculate::LogisticRegressionHypothesis(weights[j], dataEntry.features);
					if (probability > maxProbability) {
						maxProbability = probability;
						predictedTarget = j;
					}
				}
				{
					// Display result
					std::lock_guard<std::mutex> lock(ImageUtils::mutex);
					ImageUtils::numComplete++;
					if (targetOneHot[predictedTarget] == 1) {
						std::cout << "\033[" << 1 << ";0H";
						std::cout << "\r\033[K" << "\033[32m" << " VALID : " << ImageUtils::progress << "\033[0m ";
						std::cout << ((double)ImageUtils::progress / ImageUtils::numComplete) * 100.0 << std::flush;
						ImageUtils::progress++;
					}
					else {
						std::cout << "\033[" << 2 << ";0H";
						std::cout << "\r\033[K" << "\033[31m" << " WRONG : " << ImageUtils::numComplete - ImageUtils::progress << "\033[0m ";
						std::cout << (1.0 - ((double)ImageUtils::progress / ImageUtils::numComplete)) * 100.0 << std::flush;
					}
					std::cout << "\033[" << 3 << ";0H";
				}
			}
		}});
	
}

void predictTarget(const std::string& source, const std::vector<double>& featureMeans, const std::vector<double>& featureStdDevs, const std::vector<std::vector<double>>& weights)
{
	// Get image
	cv::Mat originalImage = cv::imread(source);
	if (originalImage.empty()) {
		throw std::runtime_error("Unable to load image.");
	}

	// Get transformed images
	std::vector<cv::Mat> images;
	images.push_back(originalImage.clone());
	for (int i = 1; i <= 6; ++i) {
		std::string path = source.substr(0, source.length() - 4) + "_T" + std::to_string(i) + ".JPG";
		cv::Mat image = cv::imread(path);

		if (!image.empty()) {
			images.push_back(image);
		}
		else {
			throw std::runtime_error("Unable to load transformed image.");
		}
	}

	// Create entry
	DataEntry dataEntry;

	// Add one-hot
	size_t lastSlash = source.rfind('/');
	size_t secondLastSlash = source.rfind('/', lastSlash - 1);
	dataEntry.target = source.substr(secondLastSlash + 1, lastSlash - secondLastSlash - 1);
	size_t numTargets = 8;
	std::vector<double> targetOneHot(numTargets, 0);
	for (size_t target = 0; target < numTargets; ++target) {
		if (ModelUtils::targets[target] == dataEntry.target) {
			targetOneHot[target] = 1;
			break;
		}
	}
	// Add features 
	for (int i = 0; i < 7; i++) {
		std::vector<double> features;
		std::vector<double> color = ImageProcessing::ExtractColorCaracteristics(images[i]);
		std::vector<double> texture = ImageProcessing::ExtractTextureCaracteristics(images[i]);
		features.insert(features.end(), color.begin(), color.end());
		features.insert(features.end(), texture.begin(), texture.end());
		for (const auto feature : features) {
			dataEntry.features.push_back(feature);
		}
	}
	// Normalization z-score / Filter out standard deviation zero
	size_t numFeatures = dataEntry.features.size();
	std::vector<double> newFeatures;
	for (size_t i = 0; i < numFeatures; ++i) {
		if (featureStdDevs[i] != 0.0) {
			newFeatures.push_back((dataEntry.features[i] - featureMeans[i]) / featureStdDevs[i]);
		}
	}
	dataEntry.features = std::move(newFeatures);

	// Check accuracy
	double maxProbability = -1.0;
	size_t predictedTarget = 0;
	for (size_t j = 0; j < numTargets; ++j) {
		double probability = ModelCalculate::LogisticRegressionHypothesis(weights[j], dataEntry.features);
		if (probability > maxProbability) {
			maxProbability = probability;
			predictedTarget = j;
		}
	}
	const std::string text = "Prediction: " + ModelUtils::targets[predictedTarget];

	// Show
	cv::Mat predict(originalImage.rows + 80, static_cast<int>(originalImage.cols * 2 + 60), originalImage.type(), cv::Scalar(0, 0, 0));
	originalImage.copyTo(predict(cv::Rect(20, 20, originalImage.cols, originalImage.rows)));
	images[1].copyTo(predict(cv::Rect(predict.cols - originalImage.cols - 20, 20, originalImage.cols, originalImage.rows)));
	const double fontSize = 0.75;
	const double thickness = 1.5;
	const int textWidthText = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontSize, thickness, 0).width;
	const int xPosText = (predict.cols - textWidthText) / 2;
	const int yPosText = predict.rows - 20;
	cv::putText(predict, text, cv::Point(xPosText, yPosText), cv::FONT_HERSHEY_SIMPLEX, fontSize, cv::Scalar(0, 255, 0), thickness);
	cv::imshow("Predict", predict);
	cv::waitKey(0);
}

int main(int argc, char* argv[])
{
	try {
		if (argc != 2) {
			throw std::runtime_error("Usage: " + (std::string)argv[0] + " <source_path>");
		}
		std::string source = argv[1];

		//source = "images/test/image (550).JPG";

		std::vector<std::vector<double>> weights;
		std::vector<double> featureMeans;
		std::vector<double> featureStdDevs;
		ModelUtils::LoadModels(weights, featureMeans, featureStdDevs, "models.txt");

		if (source.length() > 4 && source.substr(source.length() - 4) == ".JPG") {
			predictTarget(source, featureMeans, featureStdDevs, weights);
		}
		else {
			// FOR TEST
			if (source.back() != '/') {
				source += "/";
			}
			processImagesInDirectory(source, featureMeans, featureStdDevs, weights);
		}
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
