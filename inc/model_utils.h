#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

struct DataEntry {
	size_t index;
	std::string target;
	std::vector<double> features;
};

class ModelUtils {
public:
	static const std::vector<std::string> targets;

	static void LoadDataFile(
		std::vector<DataEntry>& database,
		const std::string& filename);

	static void SaveDataFile(
		const std::string& filename,
		const std::vector<DataEntry>& database);

	static void NormalizationZScore(
		std::vector<DataEntry>& data,
		std::vector<double>& featureMeans,
		std::vector<double>& featureStdDevs);

	static void SetupTrainingData(
		const std::vector<DataEntry>& database,
		std::vector<std::vector<double>>& weights,
		std::vector<std::vector<double>>& trainInputs,
		std::vector<std::vector<double>>& trainTargetsOneHot);

	static std::string SaveModels(
		const std::vector<std::vector<double>>& weights,
		const std::vector<double>& featureMeans,
		const std::vector<double>& featureStdDevs);

	static void LoadModels(
		std::vector<std::vector<double>>& weights,
		std::vector<double>& featureMeans,
		std::vector<double>& featureStdDevs,
		const std::string& filename);
};

#endif
