#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork()
	: m_Layers(), m_CurrentOutput()
{
}

void NeuralNetwork::AddLayer(unsigned int numInputs, unsigned int size)
{
	m_Layers.emplace_back(size, numInputs);
}

void NeuralNetwork::ForwardProp(Eigen::MatrixXf input, Eigen::VectorXi yTrue)
{
	for (size_t i = 0; i < m_Layers.size() - 1; i++)
	{
		input = m_Layers[i].Forward(input);
		input = m_Layers[i].GetReLU().Forward(input);
	}

	Layer& lastLayer = m_Layers[m_Layers.size() - 1];
	Eigen::MatrixXf output = lastLayer.Forward(input);
	output = lastLayer.GetSoftmax().Forward(output, yTrue);
	float loss = lastLayer.GetSoftmax().CalculateLoss(yTrue);

	//std::cout << "Loss: " << loss << std::endl;
	//std::cout << "Output: " << output << std::endl;

	m_CurrentOutput = output;
}

void NeuralNetwork::BackwardProp(Eigen::VectorXi yTrue)
{
	Layer& lastLayer = m_Layers[m_Layers.size() - 1];
	Eigen::MatrixXf loss_backward = lastLayer.GetSoftmax().Backward(yTrue);
	Eigen::MatrixXf layer2Output = lastLayer.Backward(loss_backward);

	Layer& firstLayer = m_Layers[0];
	Eigen::MatrixXf layer1Activation = firstLayer.GetReLU().Backward(layer2Output);
	Eigen::MatrixXf layer1Ouput = firstLayer.Backward(layer1Activation);

	//std::cout << "Layer2 dWeights: " << lastLayer.GetdWeights() << std::endl;
	//std::cout << "Layer2 dBiases: " << lastLayer.GetdBiases() << std::endl;
	//std::cout << "Layer1 dWeights: " << firstLayer.GetdWeights() << std::endl;
	//std::cout << "Layer1 dBiases: " << firstLayer.GetdBiases() << std::endl;
}

void NeuralNetwork::Optimize(Optimizer_SGD& optimizer)
{
	for (auto& layer : m_Layers)
	{
		optimizer.UpdateParams(layer);
	}
}

float NeuralNetwork::CalculateLoss(Eigen::VectorXi yTrue)
{
	return m_Layers[m_Layers.size() - 1].GetSoftmax().CalculateLoss(yTrue);
}

float NeuralNetwork::CalculateAccuracy(Eigen::VectorXi yTrue)
{
	Eigen::VectorXi outputChoices(yTrue.size());
	for (int i = 0; i < m_CurrentOutput.rows(); i++)
	{
		float max = -1.0f;
		int index = 0;

		for (int j = 0; j < m_CurrentOutput.cols(); j++)
		{
			if (m_CurrentOutput(i, j) > max)
			{
				max = m_CurrentOutput(i, j);
				index = j;
			}
		}
		outputChoices(i) = index;
	}

	//std::cout << "Output Choices: " << outputChoices << std::endl;

	int correctCount = 0;
	for (int i = 0; i < outputChoices.size(); i++)
	{
		if (outputChoices[i] == yTrue[i]) correctCount++;
	}

	float percentCorrect = (float)correctCount / (float)outputChoices.size();
	return percentCorrect;
}

//float NeuralNetwork::CalculateLoss(Eigen::VectorXi yTrue)
//{
//	return Loss_CategoricalCrossentropy::Forward(m_CurrentOutput, yTrue);
//}

Optimizer_SGD::Optimizer_SGD(float learningRate)
	: m_LearningRate(learningRate)
{
}

void Optimizer_SGD::UpdateParams(Layer& layer)
{
	layer.UpdateParams(m_LearningRate);
}
