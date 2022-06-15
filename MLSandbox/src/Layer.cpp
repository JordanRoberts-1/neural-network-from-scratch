#include "Layer.h"
#include <iostream>
#include <algorithm>

Layer::Layer(unsigned int size, unsigned int prevSize)
	: m_Size(size), m_WeightMatrix(prevSize, size),
	m_BiasVector(size)
{
	m_WeightMatrix.setRandom();
	m_BiasVector.setRandom();
}

Eigen::MatrixXf Layer::ForwardProp(const Eigen::MatrixXf& input) const
{
	Eigen::MatrixXf result(input.rows(), m_WeightMatrix.cols());

	std::cout << "Input: " << input << std::endl;
	std::cout << "Weight Matrix: " << m_WeightMatrix << std::endl;
	std::cout << "Bias Vector: " << m_BiasVector << std::endl;

	result = input * m_WeightMatrix;

	std::cout << "Matrix multiplication result: " << result << std::endl << std::endl;

	result.rowwise() += m_BiasVector.transpose();
	//std::cout << "Before Activation: " << result << std::endl;
	result = Activation_ReLU::Forward(result);
	std::cout << "After Activation: " << result << std::endl << std::endl;
	return result;
}

Eigen::MatrixXf Activation_ReLU::Forward(Eigen::MatrixXf input)
{
	return input.unaryExpr([](float x) {return std::max(0.0f, x); });
}
