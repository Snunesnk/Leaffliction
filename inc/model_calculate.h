#ifndef MODEL_CALCULATE_H
#define MODEL_CALCULATE_H

#include <vector>
#include <unordered_map>
#include <string>

struct DataEntry;

class ModelCalculate
{
public:
	static void GenerateModels(
		std::vector<DataEntry>& database,
		std::vector<std::vector<double>>& weights,
		std::vector<double>& featureMeans,
		std::vector<double>& featureStdDevs);

	static double LogisticRegressionHypothesis(
		const std::vector<double>& weights,
		const std::vector<double>& inputs);
private:

	static std::vector<double> Accuracy(
		const std::vector<std::vector<double>>& inputs,
		const std::vector<std::vector<double>>& types,
		const std::vector<std::vector<double>>& weights);

	static double LossFunction(
		const std::vector<std::vector<double>>& inputs,
		const std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target,
		const size_t type);

	static double LossFunctionPartialDerivative(
		const std::vector<std::vector<double>>& inputs,
		std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target,
		const size_t type, const size_t j);

	static void GradientDescent(
		const std::vector<std::vector<double>>& inputs,
		std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& target,
		const size_t type);

	static void LogisticRegressionTargetsOneHotTraining(
		std::vector<std::vector<double>>& weights,
		const std::vector<std::vector<double>>& trainInputs,
		const std::vector<std::vector<double>>& validInputs,
		const std::vector<std::vector<double>>& trainTargetsOneHot,
		const std::vector<std::vector<double>>& validTargetsOneHot,
		const size_t epochs);
};

#endif


