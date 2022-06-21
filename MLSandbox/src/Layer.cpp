#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <math.h>

Layer::Layer(unsigned int size, unsigned int prevSize)
	: m_Size(size), m_WeightMatrix(prevSize, size),
	m_BiasVector(size), m_Input(), m_dInputs()
{
	m_WeightMatrix = m_WeightMatrix.setRandom() * .1f;
	m_BiasVector.setZero();
}

Eigen::MatrixXf Layer::Forward(const Eigen::MatrixXf& input)
{
	m_Input = input;

	std::cout << "Input: " << input << std::endl;
	Eigen::MatrixXf result(input.rows(), m_WeightMatrix.cols());

	result = input * m_WeightMatrix;

	result.rowwise() += m_BiasVector.transpose();
	return result;
}

Eigen::MatrixXf Layer::Backward(const Eigen::MatrixXf& dValues)
{
	m_dInputs = Activation_ReLU::Backward(m_Input, dValues);

	return m_dInputs;
}

Eigen::MatrixXf Activation_ReLU::Forward(Eigen::MatrixXf input)
{
	return input.unaryExpr([](float x) {return std::max(0.0f, x); });
}

Eigen::MatrixXf Activation_ReLU::Backward(Eigen::MatrixXf z, Eigen::MatrixXf dValues)
{
	Eigen::MatrixXf dRelu(dValues);

	for (int i = 0; i < dRelu.rows(); i++)
	{
		for (int j = 0; j < dRelu.cols(); j++)
		{
			if (z(i, j) <= 0) dRelu(i, j) = 0.0f;
		}
	}

	//Apply derivative step to each element in dValues
	return dRelu;
}

Eigen::MatrixXf Activation_SoftMax::Forward(Eigen::MatrixXf input)
{
	Eigen::MatrixXf result(input.rows(), input.cols());
	for (int i = 0; i < input.rows(); i++)
	{
		Eigen::VectorXf row = input.row(i);
		float rowMax = row.maxCoeff();
		Eigen::VectorXf rowSubtracted = row.array() - rowMax;

		Eigen::VectorXf expValues(row.size());
		for (int j = 0; j < expValues.size(); j++)
		{
			expValues[j] = std::exp(rowSubtracted[j]);
		}

		float sum = expValues.sum();
		for (int j = 0; j < expValues.size(); j++)
		{
			result(i, j) = expValues[j] / sum;
		}
	}
	return result;
}

float Loss_CategoricalCrossentropy::Forward(Eigen::MatrixXf y_pred, Eigen::VectorXi yTrue)
{
	Eigen::VectorXf sampleLosses(y_pred.rows());

	for (int i = 0; i < y_pred.rows(); i++)
	{
		for (int j = 0; j < y_pred.cols(); j++)
		{
			//clip the values so that there is no divide by zero chance later
			if (y_pred(i, j) < 0.0000007f) y_pred(i, j) = 0.0000007f;
			else if (y_pred(i, j) > 1.0f - 0.0000007f) y_pred(i, j) = 1.0f - 0.0000007f;
		}
	}

	for (int i = 0; i < y_pred.rows(); i++)
	{
		sampleLosses(i) = y_pred(i, yTrue(i));
	}

	sampleLosses = sampleLosses.array().log();
	sampleLosses *= -1;
	return sampleLosses.mean();
}

float Loss_CategoricalCrossentropy::Backward(Eigen::MatrixXf dValues, Eigen::VectorXi yTrue)
{
	int samples = dValues.rows();
	int labels = dValues.cols();

	//convert yTrue into One-Hot encoded
	Eigen::MatrixXf yTrueMat(yTrue.size(), labels);
	yTrueMat.setZero();

	for (int i = 0; i < yTrueMat.rows(); i++)
	{
		yTrueMat(i, yTrue[i]) = 1.0f;
	}

	std::cout << "Y-True matrix form: " << yTrueMat << std::endl;

	//Eigen::MatrixXf result = -1.0f * yTrueMat / dValues;

	return 0.0f;
}
