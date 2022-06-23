#include "Layer.h"
#include <iostream>
#include <algorithm>
#include <math.h>

Layer::Layer(unsigned int size, unsigned int prevSize)
	: m_Size(size), m_WeightMatrix(prevSize, size),
	m_BiasVector(size), m_Input(), m_dInputs(), m_dBiases(), m_dWeights()
{
	m_WeightMatrix = m_WeightMatrix.setRandom() * 0.1f;
	m_BiasVector.setZero();
}

Eigen::MatrixXf Layer::Forward(const Eigen::MatrixXf& input)
{
	m_Input = input;
	m_Output = input * m_WeightMatrix;
	m_Output.rowwise() += m_BiasVector.transpose();

	return m_Output;
}

Eigen::MatrixXf Layer::Backward(const Eigen::MatrixXf& dActivation)
{
	m_dWeights = m_Input.transpose() * dActivation;
	m_dBiases = dActivation.colwise().sum();
	m_dInputs = dActivation * m_WeightMatrix.transpose();

	return m_dInputs;
}

void Layer::UpdateParams(float learningRate)
{
	m_WeightMatrix += -learningRate * m_dWeights;
	m_BiasVector += -learningRate * m_dBiases.row(0);
}

Eigen::MatrixXf Activation_ReLU::Forward(Eigen::MatrixXf input)
{
	m_Input = input;
	input = input.unaryExpr([](float x) {return std::max(0.0f, x); });
	m_Output = input;

	return m_Output;
}

Eigen::MatrixXf Activation_ReLU::Backward(Eigen::MatrixXf dInputs)
{
	m_dInputs = dInputs;

	for (int i = 0; i < m_dInputs.rows(); i++)
	{
		for (int j = 0; j < m_dInputs.cols(); j++)
		{
			if (m_Input(i, j) <= 0) m_dInputs(i, j) = 0.0f;
		}
	}

	//Apply derivative step to each element in dValues
	return m_dInputs;
}

Eigen::MatrixXf Activation_SoftMax_Loss_CategoricalCrossentropy::Forward(Eigen::MatrixXf input, Eigen::VectorXi yTrue)
{
	m_Inputs = input;

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
	m_Output = result;
	return result;
}

//!Fix this
Eigen::MatrixXf Activation_SoftMax_Loss_CategoricalCrossentropy::Backward(Eigen::MatrixXf output, Eigen::VectorXi y_true)
{
	int samples = output.rows();

	Eigen::MatrixXf result = output;

	for (int i = 0; i < result.rows(); i++)
	{
		result(i, y_true[i]) -= 1;
	}

	m_dInputs = result / samples;

	return m_dInputs;
}

Eigen::VectorXf Loss_CategoricalCrossentropy::Forward(Eigen::MatrixXf y_pred, Eigen::VectorXi yTrue)
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
	return sampleLosses;
}

float Loss_CategoricalCrossentropy::CalculateLoss(Eigen::MatrixXf output, Eigen::VectorXi yTrue)
{
	Eigen::VectorXf sampleLosses = Forward(output, yTrue);
	return sampleLosses.mean();
}
