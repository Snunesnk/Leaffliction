#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct DataInfo {
	std::vector<std::string> labels;
	std::vector<double> features;
	size_t index;
};

class ModelUtils {
public:
	static const std::vector<std::string> types;

	static void LoadDataFile(std::vector<DataInfo>& datainfos, const std::string& filename);
	static void SaveDataFile(const std::string& filename, const std::vector<std::vector<std::string>>& data);
	static void StandardNormalizationData(std::vector<DataInfo>& data, std::vector<double>& featureMeans, std::vector<double>& featureStdDevs);
	static void SaveModels(const std::vector<std::vector<double>>& weights, const std::vector<double>& featureMeans, const std::vector<double>& featureStdDevs, const std::string& filename);
	static void LoadModelInformations(std::vector<std::vector<double>>& weights, std::vector<double>& featureMeans, std::vector<double>& featureStdDevs, const std::string& filename);
};

#endif
