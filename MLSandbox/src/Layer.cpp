#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <math.h>

Layer::Layer(unsigned int size, unsigned int prevSize)
	: m_Size(size), m_WeightMatrix(prevSize, size),
	m_BiasVector(size)
{
	m_WeightMatrix = m_WeightMatrix.setRandom() * .1f;
	m_BiasVector.setZero();
}

Eigen::MatrixXf Layer::ForwardProp(const Eigen::MatrixXf& input) const
{
	Eigen::MatrixXf result(input.rows(), m_WeightMatrix.cols());

	result = input * m_WeightMatrix;

	result.rowwise() += m_BiasVector.transpose();
	return result;
}

Eigen::MatrixXf Activation_ReLU::Forward(Eigen::MatrixXf input)
{
	return input.unaryExpr([](float x) {return std::max(0.0f, x); });
}

Eigen::MatrixXf Activation_SoftMax::Forward(Eigen::MatrixXf input)
{
	Eigen::MatrixXf result(input.rows(), input.cols());
	for (size_t i = 0; i < input.rows(); i++)
	{
		Eigen::VectorXf row = input.row(i);
		float rowMax = row.maxCoeff();
		Eigen::VectorXf rowSubtracted = row.array() - rowMax;

		Eigen::VectorXf expValues(row.size());
		for (size_t j = 0; j < expValues.size(); j++)
		{
			expValues[j] = std::exp(rowSubtracted[j]);
		}

		float sum = expValues.sum();
		for (size_t j = 0; j < expValues.size(); j++)
		{
			result(i, j) = expValues[j] / sum;
		}
	}
	return result;
}
