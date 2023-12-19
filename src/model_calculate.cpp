#include "model_utils.h"
#include "model_calculate.h"
#include <iostream>
#include <random>
#include <iomanip>


// Function definition for generating scatter plot matrix
void extensionScatterPlotMatrix(const std::vector<DataInfo>& dataInfo, const size_t featuresCount)
{
	// Python script file name
	std::string pythonScript = "scatterplot.py";

	// Open Python script file
	std::ofstream pythonFile(pythonScript);
	if (pythonFile.is_open()) {
		// Write Python script header
		pythonFile << "import numpy as np\n";
		pythonFile << "import matplotlib.pyplot as plt\n\n";

		// Create a matrix of features values
		std::vector<std::vector<double>> featuresValues(featuresCount);
		for (size_t i = 0; i < featuresCount; ++i) {
			for (const auto& data : dataInfo) {
				featuresValues[i].push_back(data.features[i]);
			}
		}

		// Define type indices and colors

		std::vector<std::string> typeColors = { "green", "blue", "scarlet", "yellow", "cyan", "magenta", "black", "gray" };

		// Loop over types to create data arrays
		for (size_t h = 0; h < ModelUtils::types.size(); ++h) {
			std::string type = ModelUtils::types[h];
			pythonFile << "features" << h << " = np.array([";
			for (size_t k = 0; k < dataInfo.size(); ++k) {
				// Filter data for the specific type
				if (type != dataInfo[k].labels[0]) {
					continue;
				}
				// Write feature values to the array
				pythonFile << "[";
				for (size_t i = 0; i < featuresCount; ++i) {
					double featureValue = featuresValues[i][k];
					if (!std::isnan(featureValue)) {
						pythonFile << featureValue;
					}
					else {
						pythonFile << "np.nan";
					}
					if (i < featuresCount - 1) {
						pythonFile << ", ";
					}
				}
				pythonFile << "]";
				if (k < dataInfo.size() - 1) {
					pythonFile << ", ";
				}
			}
			pythonFile << "])\n";
		}

		// Set up parameters for the scatter plot matrix
		const double subplotsSizeX = 1;
		const double subplotsSizeY = 0.5;
		const double pointSize = 0.1;
		const std::string graphTitle = "Scatter Plot Matrix";

		// Write code for creating scatter plot matrix
		pythonFile << "fig, axs = plt.subplots(ncols=" << featuresCount << ", nrows=" << featuresCount << ", figsize=(" << subplotsSizeX * (double)(featuresCount) << ", " << subplotsSizeY * (double)(featuresCount) << "))\n";
		pythonFile << "fig.suptitle('" << graphTitle << "')\n";
		pythonFile << "plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.95, hspace=0.05, wspace=0.05)\n";

		// Add labels to the plots
		for (size_t feature1 = 0; feature1 < featuresCount; ++feature1) {
			pythonFile << "axs[" << feature1 << ", 0].text(0.0, 0.5, 'F " << (feature1 + 1) << "', transform=axs[" << feature1 << ", 0].transAxes, rotation=0, va='center', ha='right')\n";
		}

		for (size_t feature2 = 0; feature2 < featuresCount; ++feature2) {
			pythonFile << "axs[0, " << feature2 << "].text(0.5, 1.0, 'F " << (feature2 + 1) << "', transform=axs[0, " << feature2 << "].transAxes, va='bottom', ha='center')\n";
		}

		// Plot scatter plots for each type
		for (size_t h = 0; h < 4; ++h) {
			for (size_t feature1 = 0; feature1 < featuresCount; ++feature1) {
				for (size_t feature2 = 0; feature2 < featuresCount; ++feature2) {
					// Scatter plot for each combination of features
					pythonFile << "axs[" << feature1 << ", " << feature2 << "].scatter(features" << h << "[:, " << feature1 << "], features" << h << "[:, " << feature2 << "], marker = 'o', s = " << pointSize << ")\n";
				}
			}
		}

		// Remove ticks and labels
		pythonFile << "for ax in axs.flat:\n";
		pythonFile << "    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)\n";

		// Display the plot
		pythonFile << "plt.show()\n";
		pythonFile.close();
	}
	else {
		// Handle file writing error
		std::cerr << "Error writing Python file." << std::endl;
		return;
	}

	// Execute Python script based on the platform
	if (system(("python " + pythonScript + " &").c_str()) != 0) {
		throw std::runtime_error("Error: Failed to execute the Python command.");
	}


	// Remove the temporary Python script file
	if (std::remove(pythonScript.c_str()) != 0) {
		std::cerr << "Error deleting the temporary Python file." << std::endl;
		return;
	}
}

double ModelCalculate::Mean(const std::vector<double>& data)
{
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

double ModelCalculate::StandardDeviation(const std::vector<double>& data)
{
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

double ModelCalculate::Accuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& types, const std::vector<std::vector<double>>& weights)
{
	const size_t dataSize = inputs.size();
	const size_t typeCount = weights.size();
	double correctPredictions = 0;
	for (size_t i = 0; i < dataSize; ++i) {
		double maxProbability = -1.0;
		size_t predictedHouse = 0;
		for (size_t type = 0; type < typeCount; ++type) {
			double probability = ModelCalculate::LogisticRegressionHypothesis(weights[type], inputs[i]);
			if (probability > maxProbability) {
				maxProbability = probability;
				predictedHouse = type;
			}
		}
		// test only type accuracy
		//auto type1 = types[i][0] + types[i][1] + types[i][2] + types[i][3];
		//auto type2 = types[i][4] + types[i][5] + types[i][6] + types[i][7];
		//if ((predictedHouse < 4 && type1) || (predictedHouse >= 4 && type2)) {
		//	correctPredictions++;
		//}
		if (types[i][predictedHouse] == 1.0) {
			correctPredictions++;
		}
	}
	return (correctPredictions / static_cast<double>(dataSize)) * 100.0;
}

double ModelCalculate::LogisticRegressionHypothesis(const std::vector<double>& weights, const std::vector<double>& inputs)
{
	const size_t weightCount = weights.size();
	double weightedSum = 0;
	for (size_t i = 0; i < weightCount; ++i) {
		weightedSum += weights[i] * inputs[i];
	}
	double sigmoid = 1.0 / (1.0 + exp(-weightedSum));
	return sigmoid;
}

double ModelCalculate::LossFunction(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t type)
{
	const size_t size = inputs.size();
	double loss = 0;
	for (size_t i = 0; i < size; ++i) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[type], inputs[i]);
		loss += target[i][type] * std::log(proba + 1e-15) +
			(1.0 - target[i][type]) * std::log(1.0 - proba + 1e-15);
	}
	return -(1.0 / size) * loss;
}

double ModelCalculate::LossFunctionPartialDerivative(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t type, const size_t j)
{
	const size_t size = inputs.size();
	double derivative = 0;
	for (size_t i = 0; i < size; i++) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[type], inputs[i]);
		derivative += (proba - target[i][type]) * inputs[i][j];
	}
	return (1.0 / size) * derivative;
}

void ModelCalculate::GradientDescent(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t type)
{
	const double learningRate = 0.9;
	const size_t size = weights[0].size();
	std::vector<std::vector<double>> tmp_weights = weights;
	tmp_weights[type][0] = weights[type][0];
	for (size_t j = 0; j < size; j++) {
		double derivative = ModelCalculate::LossFunctionPartialDerivative(inputs, weights, target, type, j);

		tmp_weights[type][j] -= learningRate * derivative;
	}
	// Mise à jour des poids après avoir calculé toutes les dérivées partielles
	weights[type] = tmp_weights[type];
}

void ModelCalculate::LogisticRegressionOneHotTrainning(
	std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& trainingInputs,
	const std::vector<std::vector<double>>& validationInputs,
	const std::vector<std::vector<double>>& trainingOneHot,
	const std::vector<std::vector<double>>& validationOneHot,
	const size_t epochs)
{
	const size_t typesCount = weights.size();

	// Header
	std::cout << std::left << std::setw(std::to_string(epochs).length() + 8) << "Epochs" << std::setw(10);
	for (auto counter = 1; counter <= typesCount; counter++) {
		std::cout << "Loss " + std::to_string(counter) << std::setw(10);
	}
	std::cout << std::setw(10) << "Accuracy" << std::endl;

	// Training
	for (size_t epoch = 0; epoch < epochs; ++epoch) {
		for (size_t type = 0; type < typesCount; type++) {
			ModelCalculate::GradientDescent(trainingInputs, weights, trainingOneHot, type);
		}
		// Loss
		std::cout << "Epoch " << std::left << std::setw(std::to_string(epochs).length() + 2) << epoch + 1;
		for (size_t type = 0; type < typesCount; type++) {
			double loss = ModelCalculate::LossFunction(trainingInputs, weights, trainingOneHot, type);
			std::cout << std::setw(10) << std::setprecision(6) << loss;
		}
		double accuracy = ModelCalculate::Accuracy(validationInputs, validationOneHot, weights);
		std::cout << std::setw(5) << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
	}
}

void ModelCalculate::SetupTrainingData(
	const std::vector<DataInfo>& dataBase,
	std::vector<std::vector<double>>& weights,
	std::vector<std::vector<double>>& trainingInputs,
	std::vector<std::vector<double>>& trainingOneHot)
{
	const size_t targetCount = ModelUtils::types.size();
	const size_t featureCount = dataBase[0].features.size();

	// Init weights
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> distribution(-0.1, 0.1);
	for (size_t i = 0; i < targetCount; ++i) {
		for (size_t j = 0; j < featureCount; ++j) {
			weights[i][j] = distribution(gen);
		}
	}

	for (size_t i = 0; i < dataBase.size(); i++) {
		// Add training input
		std::vector<double> selection;
		for (size_t j = 0; j < featureCount; ++j) {
			selection.push_back(dataBase[i].features[j]);
		}
		trainingInputs.push_back(selection);
		// Add training one-hot
		std::vector<double> result(targetCount, 0.0);
		for (size_t j = 0; j < targetCount; ++j) {
			if (ModelUtils::types[j] == dataBase[i].labels[0]) {
				result[j] = 1.0;
				break;
			}
		}
		trainingOneHot.push_back(result);
	}
}

void ModelCalculate::CreateModel(std::vector<DataInfo>& dataBase)
{
	try {
		std::vector<double> featureMeans, featureStdDevs;
		ModelUtils::StandardNormalizationData(dataBase, featureMeans, featureStdDevs);

		//std::random_device rd;
		//std::mt19937 gen(rd());
		//std::shuffle(dataBase.begin(), dataBase.end(), gen);

		//extensionScatterPlotMatrix(dataBase, selectedFeatures.size());

		std::vector<std::vector<double>> weights(ModelUtils::types.size(), std::vector<double>(dataBase[0].features.size(), 0.0));
		std::vector<std::vector<double>> trainingInputs, trainingOneHot, validationInputs, validationOneHot;
		ModelCalculate::SetupTrainingData(dataBase, weights, trainingInputs, trainingOneHot);

		// Split data for validation
		std::cout << "trainingInputs : " << trainingInputs.size() << std::endl;
		std::cout << "trainingOneHot : " << trainingInputs.size() << std::endl;
		auto classSize = trainingInputs.size() / 8;
		auto forValidation = 13;
		for (int c = 1; c <= 8; c++) {
			auto lastIndexOfClass = classSize * c;
			validationInputs.insert(validationInputs.end(), trainingInputs.begin() + lastIndexOfClass - forValidation, trainingInputs.begin() + lastIndexOfClass);
			validationOneHot.insert(validationOneHot.end(), trainingOneHot.begin() + lastIndexOfClass - forValidation, trainingOneHot.begin() + lastIndexOfClass);
		}
		for (int c = 1; c <= 8; c++) {
			auto lastIndexOfClass = classSize * c - forValidation * (c - 1);
			trainingInputs.erase(trainingInputs.begin() + lastIndexOfClass - forValidation, trainingInputs.begin() + lastIndexOfClass);
			trainingOneHot.erase(trainingOneHot.begin() + lastIndexOfClass - forValidation, trainingOneHot.begin() + lastIndexOfClass);
		}
		std::cout << "for train : " << trainingInputs.size() << std::endl;
		std::cout << "for valid : " << validationInputs.size() << std::endl;

		// Train models
		ModelCalculate::LogisticRegressionOneHotTrainning(weights, trainingInputs, validationInputs, trainingOneHot, validationOneHot, 2000);

		// Save models
		ModelUtils::SaveModels(weights, featureMeans, featureStdDevs, "models.save");
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return;
	}
}
