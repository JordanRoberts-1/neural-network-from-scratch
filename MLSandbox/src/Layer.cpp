#include "Layer.h"
#include <iostream>

Layer::Layer(unsigned int size, unsigned int prevSize, bool isInput)
	: m_Size(size), m_Neurons(), m_WeightMatrix(size, prevSize),
	m_BiasVector(size), m_IsInput(isInput)
{
	//Add all the neurons
	m_Neurons.reserve(size);
	for (size_t i = 0; i < size; i++)
	{
		m_Neurons.emplace_back(prevSize);
	}

	//If this is the input layer, then no need to make a weight matrix
	if (isInput) { return; }

	//Update the weight matrix based on the neurons created above
	for (size_t i = 0; i < m_Neurons.size(); i++)
	{
		m_WeightMatrix.row(i) = m_Neurons[i].GetWeights();
		m_BiasVector[i] = m_Neurons[i].GetBias();
	}
}

Eigen::MatrixXf Layer::CalculateOutput(const Eigen::MatrixXf& input) const
{
	int numNeurons = m_WeightMatrix.rows();
	Eigen::MatrixXf result(input.rows(), numNeurons);

	result = input * m_WeightMatrix.transpose();

	std::cout << "Matrix multiplication result: " << result << std::endl;

	result.rowwise() += m_BiasVector.transpose();

	std::cout << result << std::endl;
	return result;
}
