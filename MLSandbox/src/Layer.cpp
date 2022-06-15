#include "Layer.h"
#include <iostream>

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

	std::cout << input << std::endl;
	std::cout << m_WeightMatrix << std::endl;
	std::cout << m_BiasVector << std::endl;

	result = input * m_WeightMatrix;

	std::cout << "Matrix multiplication result: " << result << std::endl << std::endl;

	result.rowwise() += m_BiasVector.transpose();

	std::cout << result << std::endl;
	return result;
}
