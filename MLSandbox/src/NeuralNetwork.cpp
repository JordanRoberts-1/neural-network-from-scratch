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

void NeuralNetwork::ForwardProp(Eigen::MatrixXf input)
{
	std::cout << "this was called" << std::endl;
	for (size_t i = 0; i < m_Layers.size() - 1; i++)
	{
		input = m_Layers[i].ForwardProp(input);
		input = Activation_ReLU::Forward(input);
	}

	Eigen::MatrixXf result = m_Layers[m_Layers.size() - 1].ForwardProp(input);
	result = Activation_SoftMax::Forward(result);

	std::cout << "FINAL RESULT: " << result << std::endl;
	m_CurrentOutput = result;
}

float NeuralNetwork::CalculateLoss(Eigen::VectorXi yTrue)
{
	return Loss_CategoricalCrossentropy::Forward(m_CurrentOutput, yTrue);
}
