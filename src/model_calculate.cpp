#include "model_utils.h"
#include "model_calculate.h"

#include <iostream>
#include <random>
#include <iomanip>
#include <execution>
#include <unordered_set>

std::vector<double> ModelCalculate::Accuracy(
	const std::vector<std::vector<double>>& inputs,
	const std::vector<std::vector<double>>& targetsOneHot,
	const std::vector<std::vector<double>>& weights)
{
	const size_t numEntries = inputs.size();
	const size_t numTargets = weights.size();

	std::vector<double> classCount(numTargets, 0);
	std::vector<double> classPrediction(numTargets, 0);
	double correctPredictions = 0;
	for (size_t i = 0; i < numEntries; ++i) {
		double maxProbability = -1.0;
		size_t predictedTarget = 0;
		for (size_t j = 0; j < numTargets; ++j) {
			double probability = ModelCalculate::LogisticRegressionHypothesis(weights[j], inputs[i]);
			if (probability > maxProbability) {
				maxProbability = probability;
				predictedTarget = j;
			}
		}
		// for class
		for (size_t j = 0; j < numTargets; ++j) {
			if (targetsOneHot[i][j] == 1) {
				classCount[j]++;
				if (predictedTarget == j) {
					classPrediction[j]++;
				}
				break;
			}
		}
		// for all
		if (targetsOneHot[i][predictedTarget] == 1.0) {
			correctPredictions++;
		}
	}
	std::vector<double> results;
	for (size_t j = 0; j < numTargets; ++j) {
		results.push_back((classPrediction[j] / classCount[j]) * 100.0);
	}
	results.push_back((correctPredictions / static_cast<double>(numEntries)) * 100.0);
	return results;
}

double ModelCalculate::LossFunction(
	const std::vector<std::vector<double>>& inputs,
	const std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& targetsOneHot,
	const size_t target)
{
	const size_t size = inputs.size();
	double loss = 0;
	for (size_t i = 0; i < size; ++i) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[target], inputs[i]);
		loss += targetsOneHot[i][target] * std::log(proba + 2.2250738585072014e-308) +
			(1.0 - targetsOneHot[i][target]) * std::log(1.0 - proba + 2.2250738585072014e-308);
	}
	return -(1.0 / size) * loss;
}

double ModelCalculate::LogisticRegressionHypothesis(
	const std::vector<double>& weights,
	const std::vector<double>& inputs)
{
	const size_t weightCount = weights.size();
	double weightedSum = 0;
	for (size_t i = 0; i < weightCount; ++i) {
		weightedSum += weights[i] * inputs[i];
	}
	double sigmoid = 1.0 / (1.0 + exp(-weightedSum));
	return sigmoid;
}

double ModelCalculate::LossFunctionPartialDerivative(
	const std::vector<std::vector<double>>& inputs,
	std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& targetsOneHot,
	const size_t target,
	const size_t j)
{
	const size_t size = inputs.size();
	double derivative = 0;
	for (size_t i = 0; i < size; i++) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[target], inputs[i]);
		derivative += (proba - targetsOneHot[i][target]) * inputs[i][j];
	}
	return (1.0 / size) * derivative;
}

void ModelCalculate::GradientDescent(
	const std::vector<std::vector<double>>& inputs,
	std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& targetsOneHot,
	const size_t target)
{
	const double learningRate = 0.1;
	const size_t size = weights[0].size();
	std::vector<std::vector<double>> tmp_weights = weights;
	tmp_weights[target][0] = weights[target][0];
	for (size_t j = 0; j < size; j++) {
		double derivative = ModelCalculate::LossFunctionPartialDerivative(inputs, weights, targetsOneHot, target, j);

		tmp_weights[target][j] -= learningRate * derivative;
	}
	// Update weights after calculating all partial derivatives
	weights[target] = tmp_weights[target];
}

void ModelCalculate::LogisticRegressionTargetsOneHotTraining(
	std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& trainInputs,
	const std::vector<std::vector<double>>& validInputs,
	const std::vector<std::vector<double>>& trainTargetsOneHot,
	const std::vector<std::vector<double>>& validTargetsOneHot,
	const size_t epochs)
{
	const size_t numTargets = weights.size();
	const std::vector<int> targetIndices = { 0, 1, 2, 3, 4, 5, 6, 7 };
	// Header
	std::cout << std::left << std::setw(std::to_string(epochs).length() + 8) << "Epochs";
	for (auto counter = 1; counter <= numTargets; counter++) {
		std::cout << std::setw(10) << "Class " + std::to_string(counter);
	}
	std::cout << std::endl;
	// Training
	for (size_t epoch = 0; epoch < epochs; ++epoch) {

		std::for_each(std::execution::par, targetIndices.begin(), targetIndices.end(), [&](int target) {
			ModelCalculate::GradientDescent(trainInputs, weights, trainTargetsOneHot, target);
			});

		// Loss
		std::cout << "Epoch " << std::left << std::setw(std::to_string(epochs).length() + 2) << epoch + 1;
		for (size_t target = 0; target < numTargets; target++) {
			double loss = ModelCalculate::LossFunction(trainInputs, weights, trainTargetsOneHot, target);
			std::cout << std::setw(10) << std::setprecision(6) << loss;
		}
		std::cout << std::endl;
		std::cout << std::left << std::setw(std::to_string(epochs).length() + 8) << "Accuracy" << std::setw(10);
		std::vector<double> accuracy = ModelCalculate::Accuracy(validInputs, validTargetsOneHot, weights);
		for (const auto value : accuracy) {
			std::cout << std::setw(10) << (std::ostringstream() << std::fixed << std::setprecision(2) << value << '%').str();
		}
		std::cout << std::endl;
	}
}

void ModelCalculate::GenerateModels(
	std::vector<DataEntry>& database,
	std::vector<std::vector<double>>& weightsAfterTraining,
	std::vector<double>& featureMeans,
	std::vector<double>& featureStdDevs)
{
	try {
		std::cout << "\r\033[K" << "Models training..." << std::endl;
		auto featuresBeforeFilter = database[0].features.size();
		ModelUtils::NormalizationZScore(database, featureMeans, featureStdDevs);
		int featuresAfterFilter = database[0].features.size();
		int featuresRemoved = featuresBeforeFilter - featuresAfterFilter;
		std::cout << "Number of features after filtering: " << featuresAfterFilter << " (" << featuresRemoved << " has been removed)" << std::endl;
		std::vector<std::vector<double>> weights(ModelUtils::targets.size(), std::vector<double>(database[0].features.size(), 0.0));
		std::vector<std::vector<double>> trainInputs, trainTargetsOneHot, validInputs, validTargetsOneHot;
		ModelUtils::SetupTrainingData(database, weights, trainInputs, trainTargetsOneHot);
		std::cout << "Inputs : " << trainInputs.size() << std::endl;
		std::random_device rd;
		std::mt19937 gen(rd());
		const size_t numTargets = trainInputs.size() / 8;
		const size_t forValidation = numTargets / 5;
		std::unordered_set<int> selectedIndices;
		for (int c = 1; c <= 8; c++) {
			while (selectedIndices.size() < forValidation * c) {
				int randomIndex = std::uniform_int_distribution<int>(numTargets * (c - 1), numTargets * c - 1)(gen);
				selectedIndices.insert(randomIndex);
			}
		}
		std::vector<decltype(trainInputs)::value_type> newTrainInputs, newTrainTargetsOneHot;
		for (size_t i = 0; i < trainInputs.size(); ++i) {
			if (selectedIndices.find(i) == selectedIndices.end()) {
				newTrainInputs.push_back(trainInputs[i]);
				newTrainTargetsOneHot.push_back(trainTargetsOneHot[i]);
			}
			else {
				validInputs.push_back(trainInputs[i]);
				validTargetsOneHot.push_back(trainTargetsOneHot[i]);
			}
		}
		trainInputs = newTrainInputs;
		trainTargetsOneHot = newTrainTargetsOneHot;
		std::cout << "For train : " << trainInputs.size() << std::endl;
		std::cout << "For valid : " << validInputs.size() << std::endl;
		ModelCalculate::LogisticRegressionTargetsOneHotTraining(weights, trainInputs, validInputs, trainTargetsOneHot, validTargetsOneHot, 200);
		weightsAfterTraining = weights;
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return;
	}
}
