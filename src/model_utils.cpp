#include "model_utils.h"
#include "model_calculate.h"
#include <iostream>
#include <fstream>
#include <sstream>

const std::vector<std::string> ModelUtils::types = {
	"Apple_Black_rot",
	"Apple_healthy",
	"Apple_rust",
	"Apple_scab",
	"Grape_Black_rot",
	"Grape_Esca",
	"Grape_healthy",
	"Grape_spot"
};

void ModelUtils::SaveDataFile(const std::string& filename, const std::vector<std::vector<std::string>>& data)
{
	std::ofstream outputfile(filename);
	if (outputfile.is_open()) {
		for (size_t i = 0; i < data.size(); i++) {
			for (size_t j = 0; j < data[i].size(); j++) {
				outputfile << data[i][j];
				if (j < data[i].size() - 1) {
					outputfile << ",";
				}
				else {
					outputfile << std::endl;
				}
			}
		}
		outputfile.close();
		std::cout << "Data saved to " << filename << std::endl;
	}
	else {
		throw std::runtime_error("Unable to open " + filename);
	}
}

void ModelUtils::LoadDataFile(std::vector<DataInfo>& datainfos, const std::string& filename)
{
	std::ifstream inputfile(filename);

	if (!inputfile.is_open()) {
		throw std::runtime_error("Unable to open " + filename);
	}

	try {
		std::string line;
		while (std::getline(inputfile, line)) {
			std::istringstream linestream(line);
			std::string element;
			std::vector<std::string> row;
			while (std::getline(linestream, element, ',')) {
				row.push_back(element);
			}
			DataInfo datainfo;
			datainfo.index = std::stoi(row[0]);
			datainfo.labels.push_back(row[1]);
			datainfo.labels.push_back(row[2]);
			for (size_t i = 3; i < row.size(); i++) {
				datainfo.features.push_back(std::stod(row[i]));
			}
			datainfos.push_back(datainfo);
		}
	}
	catch (...) {
		throw std::runtime_error("Wrong data from " + filename);
	}

	inputfile.close();
	std::cout << "Data loaded from " << filename << std::endl;
}


void ModelUtils::StandardNormalizationData(std::vector<DataInfo>& data, std::vector<double>& featureMeans, std::vector<double>& featureStdDevs)
{
	if (featureMeans.size() + featureStdDevs.size() == 0) {
		for (size_t i = 0; i < data[0].features.size(); ++i) {
			std::vector<double> featureValues;
			for (const DataInfo& entry : data) {
				featureValues.push_back(entry.features[i]);
			}
			double mean = ModelCalculate::Mean(featureValues);
			double stdDev = ModelCalculate::StandardDeviation(featureValues);

			featureMeans.push_back(mean);
			featureStdDevs.push_back(stdDev);
		}
	}
	int warnings = 0;
	std::cout << std::endl;
	for (DataInfo& entry : data) {
		for (size_t i = 0; i < entry.features.size(); ++i) {
			if (featureStdDevs[i] != 0.0) {
				entry.features[i] = (entry.features[i] - featureMeans[i]) / featureStdDevs[i];
			}
			else {
				entry.features[i] = 0;
			}
		}
	}
	for (auto featureStdDev : featureStdDevs) {
		if (featureStdDev == 0) {
			warnings++;
		}
	}
	if (warnings) {
		std::cout << "Warning: " << warnings << " features have a zero standard deviation." << std::endl;
	}
	std::cout << std::endl;
}

void ModelUtils::SaveModelInformations(const std::vector<std::vector<double>>& weights,
	const std::vector<double>& featureMeans,
	const std::vector<double>& featureStdDevs,
	const std::string& filename)
{
	// Ouvrir le fichier en mode écriture
	std::ofstream outFile(filename);

	if (!outFile.is_open()) {
		std::cerr << "Error: Unable to open the file " << filename << " for writing." << std::endl;
		return;
	}

	// Enregistrer les caractéristiques (moyennes et écarts types)
	outFile << "FeatureMeans:";
	for (double mean : featureMeans) {
		outFile << " " << mean;
	}
	outFile << "\n";

	outFile << "FeatureStandardDeviations:";
	for (double stdDev : featureStdDevs) {
		outFile << " " << stdDev;
	}
	outFile << "\n\n";

	// Enregistrer les poids
	for (const auto& typeWeights : weights) {
		for (double weight : typeWeights) {
			outFile << weight << " ";
		}
		outFile << "\n";
	}

	// Fermer le fichier
	outFile.close();
}

void ModelUtils::LoadModelInformations(std::vector<std::vector<double>>& weights,
	std::vector<double>& featureMeans,
	std::vector<double>& featureStdDevs,
	const std::string& filename)
{
	// Ouvrir le fichier en mode lecture
	std::ifstream inFile(filename);

	if (!inFile.is_open()) {
		std::cerr << "Error: Unable to open the file " << filename << " for reading." << std::endl;
		return;
	}

	std::string line;
	// Lire les moyennes des caractéristiques
	std::getline(inFile, line);
	std::istringstream meanStream(line);
	std::string meanLabel;
	meanStream >> meanLabel; // Ignorer le label "Feature Means:"
	double meanValue;
	while (meanStream >> meanValue) {
		featureMeans.push_back(meanValue);
	}

	// Lire les écarts types des caractéristiques
	std::getline(inFile, line);
	std::istringstream stdDevStream(line);
	std::string stdDevLabel;
	stdDevStream >> stdDevLabel; // Ignorer le label "Feature Standard Deviations:"
	double stdDevValue;
	while (stdDevStream >> stdDevValue) {
		featureStdDevs.push_back(stdDevValue);
	}

	// Lire les poids
	weights.clear(); // Assurez-vous de vider le vecteur avant de le remplir
	std::getline(inFile, line);
	while (std::getline(inFile, line)) {
		std::istringstream weightStream(line);
		double weight;
		std::vector<double> typeWeights;
		while (weightStream >> weight) {
			typeWeights.push_back(weight);
		}
		weights.push_back(typeWeights);
	}

	// Fermer le fichier
	inFile.close();
}