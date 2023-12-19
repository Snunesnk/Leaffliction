#ifndef MODEL_CALCULATE_H
#define MODEL_CALCULATE_H

#include <vector>
#include <unordered_map>
#include <string>

struct DataInfo;

class ModelCalculate {
public:
	static double Mean(const std::vector<double>& data);

	static double StandardDeviation(const std::vector<double>& data);

	static double LogisticRegressionHypothesis(const std::vector<double>& weights, const std::vector<double>& inputs);

	static double Accuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& types,
		const std::vector<std::vector<double>>& weights);

	static double LossFunction(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target, const size_t type);

	static double LossFunctionPartialDerivative(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target, const size_t type, const size_t j);

	static void GradientDescent(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target, const size_t type);

	static void LogisticRegressionOneHotTrainning(
		std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& trainingInputs,
		const std::vector<std::vector<double>>& validationInputs,
		const std::vector<std::vector<double>>& trainingOneHot,
		const std::vector<std::vector<double>>& validationOneHot,
		const size_t epochs);

	static void SetupTrainingData(const std::vector<DataInfo>& dataBase, std::vector<std::vector<double>>& weights,
		std::vector<std::vector<double>>& trainingInputs, std::vector<std::vector<double>>& trainingOneHot);
	
	static void CreateModel(std::vector<DataInfo>& dataBase);
};

#endif


