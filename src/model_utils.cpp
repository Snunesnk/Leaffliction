#include "model_utils.h"

#include <iostream>

const std::vector<std::string> ModelUtils::targets =
{
	"Apple_Black_rot",
	"Apple_healthy",
	"Apple_rust",
	"Apple_scab",
	"Grape_Black_rot",
	"Grape_Esca",
	"Grape_healthy",
	"Grape_spot"
};

void ModelUtils::SaveDataFile(
	const std::string& filename,
	const std::vector<DataEntry>& database)
{
	std::ofstream outputfile(filename);
	if (outputfile.is_open()) {
		for (size_t i = 0; i < database.size(); i++) {
			outputfile << database[i].index << ",";
			outputfile << database[i].target << ",";
			for (size_t j = 0; j < database[i].features.size(); j++) {
				outputfile << std::fixed << database[i].features[j];
				if (j < database[i].features.size() - 1) {
					outputfile << ",";
				}
				else {
					outputfile << std::endl;
				}
			}
		}
		outputfile.close();
	}
	else {
		throw std::runtime_error("Unable to open " + filename);
	}
}

void ModelUtils::LoadDataFile(
	std::vector<DataEntry>& database,
	const std::string& filename)
{
	std::cout << "\r\033[K" << "Loading data.csv..." << std::endl;
	std::ifstream inputfile(filename);

	if (!inputfile.is_open()) {
		throw std::runtime_error("Unable to open " + filename);
	}

	try {
		std::string entry;
		while (std::getline(inputfile, entry)) {
			std::istringstream linestream(entry);
			std::string element;
			std::vector<std::string> row;
			while (std::getline(linestream, element, ',')) {
				row.push_back(element);
			}
			DataEntry entry;
			entry.index = std::stoi(row[0]);
			entry.target = row[1];
			for (size_t i = 2; i < row.size(); i++) {
				entry.features.push_back(std::stod(row[i]));
			}
			database.push_back(entry);
		}
	}
	catch (...) {
		throw std::runtime_error("Wrong data from " + filename);
	}

	inputfile.close();
	std::cout << "Data loaded from " << filename << std::endl;
}

void ModelUtils::NormalizationZScore(
	std::vector<DataEntry>& database,
	std::vector<double>& featureMeans,
	std::vector<double>& featureStdDevs)
{
	const size_t numFeatures = database[0].features.size();

	for (size_t i = 0; i < numFeatures; ++i) {
		std::vector<double> featureValues;
		for (const DataEntry& entry : database) {
			featureValues.push_back(entry.features[i]);
		}
		// Calculate mean
		double sum = 0.0;
		for (const double& value : featureValues) {
			sum += value;
		}
		double mean = sum / database.size();

		// Calculate standard deviation
		double sumSquaredDiff = 0.0;
		for (const double& value : featureValues) {
			double diff = value - mean;
			sumSquaredDiff += diff * diff;
		}
		double standardDeviation = std::sqrt(sumSquaredDiff / database.size());

		featureMeans.push_back(mean);
		featureStdDevs.push_back(standardDeviation);
	}

	//Filter
	for (DataEntry& entry : database) {
		std::vector<double> newFeatures;
		for (size_t i = 0; i < numFeatures; ++i) {
			// Filter out standard deviation zero
			if (featureStdDevs[i] != 0.0) {
				newFeatures.push_back((entry.features[i] - featureMeans[i]) / featureStdDevs[i]);
			}
		}
		entry.features = newFeatures;
	}
}

void ModelUtils::SetupTrainingData(
	const std::vector<DataEntry>& database,
	std::vector<std::vector<double>>& weights,
	std::vector<std::vector<double>>& trainInputs,
	std::vector<std::vector<double>>& trainTargetsTargetsOneHot)
{
	const size_t numEntries = database.size();
	const size_t numTargets = ModelUtils::targets.size();
	const size_t numFeatures = database[0].features.size();
	//// Init weights
	//std::random_device rd;
	//std::mt19937 gen(rd());
	//std::uniform_real_distribution<double> distribution(-0.0, 0.0);
	//for (size_t target = 0; target < numTargets; ++target) {
	//	for (size_t feature = 0; feature < numFeatures; ++feature) {
	//		weights[target][feature] = distribution(gen);
	//	}
	//}
	for (size_t entry = 0; entry < numEntries; entry++) {
		// Add train input
		trainInputs.push_back(database[entry].features);
		// Add train one-hot
		std::vector<double> result(numTargets, 0.0);
		for (size_t target = 0; target < numTargets; ++target) {
			if (ModelUtils::targets[target] == database[entry].target) {
				result[target] = 1.0;
				break;
			}
		}
		trainTargetsTargetsOneHot.push_back(result);
	}
}


std::string ModelUtils::SaveModels(
	const std::vector<std::vector<double>>& weights,
	const std::vector<double>& featureMeans,
	const std::vector<double>& featureStdDevs)
{
	std::ostringstream oss;

	// Mean
	for (double mean : featureMeans) {
		oss << mean << " ";
	}
	oss << "\n";
	// StdDev
	for (double stdDev : featureStdDevs) {
		oss << stdDev << " ";
	}
	oss << "\n\n";
	// Weights
	for (const auto& targetWeights : weights) {
		for (double weight : targetWeights) {
			oss << weight << " ";
		}
		oss << "\n";
	}

	return oss.str();
}


void ModelUtils::LoadModels(
	std::vector<std::vector<double>>& weights,
	std::vector<double>& featureMeans,
	std::vector<double>& featureStdDevs,
	const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << filename << std::endl;
		return;
	}

	// Clear existing data
	weights.clear();
	featureMeans.clear();
	featureStdDevs.clear();

	std::string line;

	// Load feature means
	if (std::getline(file, line)) {
		std::istringstream meanStream(line);
		double mean;
		while (meanStream >> mean) {
			featureMeans.push_back(mean);
		}
	}

	// Load feature standard deviations
	if (std::getline(file, line)) {
		std::istringstream stdDevStream(line);
		double stdDev;
		while (stdDevStream >> stdDev) {
			featureStdDevs.push_back(stdDev);
		}
	}

	// Load weights
	while (std::getline(file, line)) {
		if (line.empty()) continue;
		std::istringstream weightStream(line);
		std::vector<double> targetWeights;
		double weight;
		while (weightStream >> weight) {
			targetWeights.push_back(weight);
		}
		weights.push_back(targetWeights);
	}

	file.close();
}
