#pragma once
#include <vector>
#include "Layer.h"

class Optimizer_SGD
{
public:
	Optimizer_SGD(float learningRate);
	void UpdateParams(Layer& layer) const;
private:
	float m_LearningRate;
};

class NeuralNetwork
{
public:
	NeuralNetwork();
	void AddLayer(unsigned int numInputs, unsigned int size);
	inline Layer& GetLayer(unsigned int index) { return m_Layers[index]; }
	inline Eigen::MatrixXf GetOutput() const { return m_CurrentOutput; }

	void ForwardProp(Eigen::MatrixXf* input);
	void BackwardProp(const Eigen::MatrixXf& yTrue);
	void Fit(Eigen::MatrixXf input, const Eigen::MatrixXf& y, const Optimizer_SGD& optimizer);

	void Optimize(const Optimizer_SGD& optimizer);
	float CalculateLoss(const Eigen::MatrixXf& yTrue);
	float CalculateAccuracy(Eigen::VectorXf yTrue);

	Eigen::VectorXf GetQs(const Eigen::VectorXf& input);
	int Predict(const Eigen::VectorXf& input);

private:
	std::vector<Layer> m_Layers;
	Eigen::MatrixXf m_CurrentOutput;
};

