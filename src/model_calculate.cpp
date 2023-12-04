#include <iostream>
#include <random>
#include "model_utils.h"
#include "model_calculate.h"

double ModelCalculate::Mean(const std::vector<double>& data) {
	double sum = 0.0;
	double count = 0;
	for (const auto& value : data) {
		if (!std::isnan(value)) {
			sum += value;
			count++;
		}
	}
	return sum / count;
}

double ModelCalculate::StandardDeviation(const std::vector<double>& data) {
	double m = ModelCalculate::Mean(data);
	double variance = 0.0;
	double count = 0;
	for (const auto& value : data) {
		if (!std::isnan(value)) {
			variance += std::pow(value - m, 2);
			count++;
		}
	}
	return std::sqrt(variance / count);
}

double ModelCalculate::Min(const std::vector<double>& data) {
	double minValue = data[0];
	for (const auto& value : data) {
		if (!std::isnan(value) && value < minValue) {
			minValue = value;
		}
	}
	return minValue;
}

double ModelCalculate::Max(const std::vector<double>& data) {
	double maxValue = data[0];
	for (const auto& value : data) {
		if (!std::isnan(value) && value > maxValue) {
			maxValue = value;
		}
	}
	return maxValue;
}

double ModelCalculate::Quartile(const std::vector<double>& data, int n) {
	std::vector<double> sortedData;
	for (const auto& value : data) {
		if (!std::isnan(value)) {
			sortedData.push_back(value);
		}
	}
	std::sort(sortedData.begin(), sortedData.end());
	double index = (static_cast<double>(n) * (static_cast<double>(sortedData.size()) - 1.0)) / 100.0;
	double ptc = index - static_cast<double>(static_cast<size_t>(index));
	return (sortedData[static_cast<size_t>(index)] * ptc + sortedData[static_cast<size_t>(index) + 1] * (1.0 - ptc)) / 2.0;
}

double ModelCalculate::Covariance(const std::vector<double>& data1, const std::vector<double>& data2) {
	double meanData1 = ModelCalculate::Mean(data1);
	double meanData2 = ModelCalculate::Mean(data2);
	// Calcul de la covariance
	double covariance = 0.0;
	double count = 0;
	for (size_t i = 0; i < data1.size(); ++i) {
		if (!std::isnan(data1[i]) && !std::isnan(data2[i])) {
			covariance += (data1[i] - meanData1) * (data2[i] - meanData2);
			count++;
		}
	}
	return covariance / count;
}

double ModelCalculate::PearsonCorrelation(const std::vector<double>& data1, const std::vector<double>& data2) {
	double covariance = ModelCalculate::Covariance(data1, data2);
	double stdDevData1 = ModelCalculate::StandardDeviation(data1);
	double stdDevData2 = ModelCalculate::StandardDeviation(data2);
	// Calcul du coefficient de corrélation de Pearson
	double pearsonCorrelation = covariance / (stdDevData1 * stdDevData2);
	return pearsonCorrelation;
}

double ModelCalculate::LogisticRegressionHypothesis(const std::vector<double>& weights, const std::vector<double>& inputs) {
	const size_t weightCount = weights.size();
	double weightedSum = 0;
	for (size_t i = 0; i < weightCount; ++i) {
		weightedSum += weights[i] * inputs[i];
	}
	double sigmoid = 1.0 / (1.0 + exp(-weightedSum));
	return sigmoid;
}

double ModelCalculate::Accuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets,
	const std::vector<std::vector<double>>& weights) {
	const size_t dataSize = inputs.size();
	const size_t houseCount = weights.size();
	double correctPredictions = 0;
	for (size_t i = 0; i < dataSize; ++i) {
		double maxProbability = -1.0;
		size_t predictedHouse = 0;
		for (size_t house = 0; house < houseCount; ++house) {
			double probability = ModelCalculate::LogisticRegressionHypothesis(weights[house], inputs[i]);
			if (probability > maxProbability) {
				maxProbability = probability;
				predictedHouse = house;
			}
		}
		if (targets[i][predictedHouse] == 1.0) {
			correctPredictions++;
		}
	}
	return (correctPredictions / static_cast<double>(dataSize)) * 100.0;
}


double ModelCalculate::LossFunction(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t house) {
	const size_t size = inputs.size();
	double loss = 0;
	for (size_t i = 0; i < size; ++i) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[house], inputs[i]);
		loss += target[i][house] * std::log(proba + 1e-15) +
			(1.0 - target[i][house]) * std::log(1.0 - proba + 1e-15);
	}
	return -(1.0 / size) * loss;
}

double ModelCalculate::LossFunctionPartialDerivative(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t house, const size_t j) {
	const size_t size = inputs.size();
	double derivative = 0;
	for (size_t i = 0; i < size; i++) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[house], inputs[i]);
		derivative += (proba - target[i][house]) * inputs[i][j];
	}
	return (1.0 / size) * derivative;
}

void ModelCalculate::GradientDescent(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t house) {
	const double learningRate = 0.1;
	const size_t size = weights[0].size();
	std::vector<std::vector<double>> tmp_weights = weights;
	tmp_weights[house][0] = weights[house][0];
	for (size_t j = 0; j < size; j++) {
		double derivative = ModelCalculate::LossFunctionPartialDerivative(inputs, weights, target, house, j);

		tmp_weights[house][j] -= learningRate * derivative;
	}
	// Mise à jour des poids après avoir calculé toutes les dérivées partielles
	weights[house] = tmp_weights[house];
}

void ModelCalculate::LogisticRegressionTrainning(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& inputs,
	const std::vector<std::vector<double>>& targets, const size_t epochs) {
	const size_t housesCount = weights.size();
	std::cout << std::left << std::setw(std::to_string(epochs).length() + 8) << "Epochs"
		<< std::setw(10) << "Loss 1"
		<< std::setw(10) << "Loss 2"
		<< std::setw(10) << "Loss 3"
		<< std::setw(10) << "Loss 4"
		<< std::setw(10) << "Accuracy" << std::endl;
	// Entraînement du modèle
	for (size_t epoch = 0; epoch < epochs; ++epoch) {
		for (size_t house = 0; house < housesCount; house++) {
			ModelCalculate::GradientDescent(inputs, weights, targets, house);
		}
		// Calculer la perte moyenne pour chaque maison après chaque époque (facultatif)
		std::cout << "Epoch " << std::left << std::setw(std::to_string(epochs).length() + 2) << epoch + 1;
		for (size_t house = 0; house < housesCount; house++) {
			double loss = ModelCalculate::LossFunction(inputs, weights, targets, house);
			std::cout << std::setw(10) << std::setprecision(6) << loss;
		}
		double accuracy = ModelCalculate::Accuracy(inputs, targets, weights);
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << accuracy << "%";
		std::cout << std::endl;
	}
}

// Function to handle missing values by replacing NaN with feature means
void ModelCalculate::HandleMissingValues(std::vector<DataInfo>& datas) {
	size_t dataCount = datas.size();
	size_t featureCount = datas[0].features.size();
	std::vector<std::vector<double>> featureValues(featureCount, std::vector<double>(datas.size()));
	for (size_t i = 0; i < featureCount; ++i) {
		for (size_t j = 0; j < dataCount; ++j) {
			featureValues[i][j] = datas[j].features[i];
		}
	}
	for (auto& data : datas) {
		for (size_t j = 0; j < featureCount; ++j) {
			if (std::isnan(data.features[j])) {
				data.features[j] = ModelCalculate::Mean(featureValues[j]);
			}
		}
	}
}

// Function to set up data for training
void ModelCalculate::SetupTrainingData(const std::vector<DataInfo>& datas, const std::vector<size_t>& selectedFeatures,
	const std::unordered_map<size_t, std::string>& houseIndex, std::vector<std::vector<double>>& weights,
	std::vector<std::vector<double>>& trainingInputs, std::vector<std::vector<double>>& trainingLabels) {
	const size_t houseCount = houseIndex.size();
	// Initialize weights randomly
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distribution(-0.5, 0.5);
	for (size_t i = 0; i < houseCount; ++i) {
		for (size_t j = 0; j < selectedFeatures.size(); ++j) {
			weights[i][j] = distribution(gen);
		}
	}
	// Populate training data
	for (size_t i = 0; i < datas.size(); i++) {
		std::vector<double> selection;
		for (auto feature : selectedFeatures) {
			selection.push_back(datas[i].features[feature - 1]);
		}
		trainingInputs.push_back(selection);
		std::vector<double> result(houseCount, 0.0);
		for (const auto& entry : houseIndex) {
			if (entry.second == datas[i].labels[0]) {
				result[entry.first] = 1.0;
				break;
			}
		}
		trainingLabels.push_back(result);
	}
}

void ModelCalculate::CreateModel(std::vector<DataInfo>& datas) {
	try {
		// Handle missing values
		ModelCalculate::HandleMissingValues(datas);
		// Normalize training data
		std::vector<double> featureMeans, featureStdDevs;
		ModelUtils::NormalizeData(datas, featureMeans, featureStdDevs);
		// Set up data for training
		std::vector<size_t> selectedFeatures = { 3, 4, 7 };
		std::unordered_map<size_t, std::string> houseIndex = {
			{ 0, "Apple_Black_rot" },
			{ 1, "Apple_healthy" },
			{ 2, "Apple_rust" },
			{ 3, "Apple_scab" },
			{ 4, "Grape_Black_rot" },
			{ 5, "Grape_Esca" },
			{ 6, "Grape_healthy" },
			{ 7, "Grape_spot" }
		};
		std::vector<std::vector<double>> weights(houseIndex.size(), std::vector<double>(selectedFeatures.size(), 0.0));
		std::vector<std::vector<double>> inputs, targets;
		ModelCalculate::SetupTrainingData(datas, selectedFeatures, houseIndex, weights, inputs, targets);
		// Train the model
		ModelCalculate::LogisticRegressionTrainning(weights, inputs, targets, 100);
		// Save weights and normalization parameters
		ModelUtils::SaveWeightsAndNormalizationParameters(weights, featureMeans, featureStdDevs, "models.save");
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return;
	}
}
