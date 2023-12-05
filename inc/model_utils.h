#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <functional>

struct DataInfo {
	std::vector<std::string> labels;
	std::vector<double> features;
	size_t index;
};

class ModelUtils {
public:
	static const std::vector<std::string> targets;

	static bool isNumber(const std::string& str);
	static std::pair<std::vector<std::string>, size_t> LoadDataFile(const std::string& filename, std::vector<DataInfo>& datas);
	static void printFeatureHeader(const size_t max);
	template <typename Function>
	static void computeAndPrintFeatures(const std::string& sectionName, Function function, std::vector<std::vector<double>> featuresValues) {
		const int fieldWidth = 14; // Output field width.

		// Print section name and computed features.
		std::cout << std::setw(fieldWidth) << std::left << sectionName;
		for (const auto& value : featuresValues) {
			std::cout << std::setw(fieldWidth) << std::right << std::fixed << std::setprecision(6) << function(value);
		}
		std::cout << std::endl;
	}
	static void NormalizeData(std::vector<DataInfo>& data, std::vector<double>& featureMeans, std::vector<double>& featureStdDevs);
	static void SaveWeightsAndNormalizationParameters(const std::vector<std::vector<double>>& weights, const std::vector<double>& featureMeans, const std::vector<double>& featureStdDevs, const std::string& filename);
	static void LoadWeightsAndNormalizationParameters(std::vector<std::vector<double>>& weights, std::vector<double>& featureMeans, std::vector<double>& featureStdDevs, const std::string& filename);
};

#endif
