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
	static double Min(const std::vector<double>& data);
	static double Max(const std::vector<double>& data);
	static double Quartile(const std::vector<double>& data, int n);
	static double Covariance(const std::vector<double>& data1, const std::vector<double>& data2);
	static double PearsonCorrelation(const std::vector<double>& data1, const std::vector<double>& data2);
	static double LogisticRegressionHypothesis(const std::vector<double>& weights, const std::vector<double>& inputs);
	static double Accuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& types, const std::vector<std::vector<double>>& weights);
	static double LossFunction(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& target, const size_t type);
	static double LossFunctionPartialDerivative(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& target, const size_t type, const size_t j);
	static void GradientDescent(const std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& target, const size_t type);
	static void LogisticRegressionTrainning(std::vector<std::vector<double>>& weights, const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& types, const size_t epochs);
	static void HandleMissingValues(std::vector<DataInfo>& datas);
	static void SetupTrainingData(const std::vector<DataInfo>& datas, const std::vector<size_t>& selectedFeatures, std::vector<std::vector<double>>& weights, std::vector<std::vector<double>>& trainingInputs, std::vector<std::vector<double>>& trainingLabels);
	static void CreateModel(std::vector<DataInfo>& datas);
};

#endif


