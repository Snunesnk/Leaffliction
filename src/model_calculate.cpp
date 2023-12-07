#include <iostream>
#include <random>
#include "model_utils.h"
#include "model_calculate.h"


// Function definition for generating scatter plot matrix
void extensionScatterPlotMatrix(const std::vector<DataInfo>& dataInfo, const size_t featuresCount) {
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

double ModelCalculate::Accuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& types,
	const std::vector<std::vector<double>>& weights) {
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
		if (types[i][predictedHouse] == 1.0) {
			correctPredictions++;
		}
	}
	return (correctPredictions / static_cast<double>(dataSize)) * 100.0;
}


double ModelCalculate::LossFunction(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t type) {
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
	const std::vector<std::vector<double>>& target, const size_t type, const size_t j) {
	const size_t size = inputs.size();
	double derivative = 0;
	for (size_t i = 0; i < size; i++) {
		double proba = ModelCalculate::LogisticRegressionHypothesis(weights[type], inputs[i]);
		derivative += (proba - target[i][type]) * inputs[i][j];
	}
	return (1.0 / size) * derivative;
}

void ModelCalculate::GradientDescent(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
	const std::vector<std::vector<double>>& target, const size_t type) {
	const double learningRate = 0.1;
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

void ModelCalculate::LogisticRegressionTrainning(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& inputs,
	const std::vector<std::vector<double>>& types, const size_t epochs) {
	const size_t typesCount = weights.size();
	std::cout << std::left << std::setw(std::to_string(epochs).length() + 8) << "Epochs"
		<< std::setw(10) << "Loss 1"
		<< std::setw(10) << "Loss 2"
		<< std::setw(10) << "Loss 3"
		<< std::setw(10) << "Loss 4"
		<< std::setw(10) << "Accuracy" << std::endl;
	// Entraînement du modèle
	for (size_t epoch = 0; epoch < epochs; ++epoch) {
		for (size_t type = 0; type < typesCount; type++) {
			ModelCalculate::GradientDescent(inputs, weights, types, type);
		}
		// Calculer la perte moyenne pour chaque maison après chaque époque (facultatif)
		std::cout << "Epoch " << std::left << std::setw(std::to_string(epochs).length() + 2) << epoch + 1;
		for (size_t type = 0; type < typesCount; type++) {
			double loss = ModelCalculate::LossFunction(inputs, weights, types, type);
			std::cout << std::setw(10) << std::setprecision(6) << loss;
		}
		double accuracy = ModelCalculate::Accuracy(inputs, types, weights);
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
	std::vector<std::vector<double>>& weights, std::vector<std::vector<double>>& trainingInputs, std::vector<std::vector<double>>& trainingLabels) {
	const size_t targetCount = ModelUtils::types.size();
	// Initialize weights randomly
	std::mt19937 gen(42);
	std::uniform_real_distribution<double> distribution(-1.0, 1.0);
	for (size_t i = 0; i < targetCount; ++i) {
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
		std::vector<double> result(targetCount, 0.0);
		for (size_t j = 0; j < targetCount; ++j) {
			if (ModelUtils::types[j] == datas[i].labels[0]) {
				result[j] = 1.0;
				break;
			}
		}
		trainingLabels.push_back(result);
	}
}

void ModelCalculate::CreateModel(std::vector<DataInfo>& datas) {
	try {
		//extensionScatterPlotMatrix(datas, 8);
		// Handle missing values
		//ModelCalculate::HandleMissingValues(datas);
		// Normalize training data
		std::vector<double> featureMeans, featureStdDevs;
		ModelUtils::NormalizeData(datas, featureMeans, featureStdDevs);
		// Set up data for training
		std::vector<size_t> selectedFeatures = { 1, 2, 3, 4, 5, 6, 7, 8 };

		std::vector<std::vector<double>> weights(ModelUtils::types.size(), std::vector<double>(selectedFeatures.size(), 0.0));
		std::vector<std::vector<double>> inputs, types;
		ModelCalculate::SetupTrainingData(datas, selectedFeatures, weights, inputs, types);
		// Train the model
		ModelCalculate::LogisticRegressionTrainning(weights, inputs, types, 2000);
		// Save weights and normalization parameters
		ModelUtils::SaveWeightsAndNormalizationParameters(weights, featureMeans, featureStdDevs, "models.save");
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return;
	}
}
