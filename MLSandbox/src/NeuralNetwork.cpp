#include "NeuralNetwork.h"
#include <iostream>

NeuralNetwork::NeuralNetwork()
	: m_Layers()
{
}

void NeuralNetwork::AddLayer(unsigned int numInputs, unsigned int size)
{
	m_Layers.emplace_back(size, numInputs);
}

void NeuralNetwork::ForwardProp(Eigen::MatrixXf input)
{
	for (size_t i = 0; i < m_Layers.size(); i++)
	{
		input = m_Layers[i].ForwardProp(input);
	}

	std::cout << "FINAL RESULT: " << input << std::endl;
}
