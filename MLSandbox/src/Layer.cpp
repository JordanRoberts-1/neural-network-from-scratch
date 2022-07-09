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

Eigen::MatrixXf* Layer::Forward(const Eigen::MatrixXf& input)
{
	//Forward pass for a layer is: output = input * weights + biasVector
	m_Input = input;
	m_Output = input * m_WeightMatrix;
	m_Output.rowwise() += m_BiasVector.transpose();

	return &m_Output;
}

Eigen::MatrixXf Layer::Backward(const Eigen::MatrixXf& dActivation)
{
	//Calculate the derivative values for the weights, biases, and the inputs
	m_dWeights = m_Input.transpose() * dActivation;
	m_dBiases = dActivation.colwise().sum();
	m_dInputs = dActivation * m_WeightMatrix.transpose();

	//dInputs continues to be passed in the chain
	return m_dInputs;
}

Eigen::MatrixXf Layer::Predict(const Eigen::MatrixXf& input)
{
	Eigen::MatrixXf output = input * m_WeightMatrix;
	output.rowwise() += m_BiasVector.transpose();
	return output;
}

void Layer::UpdateParams(float learningRate)
{
	//Update the weight values and bias values based on the gradient and learning rate
	m_WeightMatrix += -learningRate * m_dWeights;
	m_BiasVector += -learningRate * m_dBiases.row(0);
}

Eigen::MatrixXf* Activation_ReLU::Forward(const Eigen::MatrixXf& input)
{
	//Apply F(x) = max(0, x) to every element in the input
	m_Input = input;
	m_Output = input.unaryExpr([](float x) {return std::max(0.0f, x); });

	return &m_Output;
}

Eigen::MatrixXf Activation_ReLU::Backward(const Eigen::MatrixXf& dInputs)
{
	m_dInputs = dInputs;

	//If the input that we started with in the forward pass was less than 0, then 
	//the dInput = 0, otherwise the value is juse the original value passed into this function
	for (int i = 0; i < m_dInputs.rows(); i++)
	{
		for (int j = 0; j < m_dInputs.cols(); j++)
		{
			if (m_Input(i, j) <= 0) m_dInputs(i, j) = 0.0f;
		}
	}

	return m_dInputs;
}

Eigen::MatrixXf Activation_ReLU::Predict(const Eigen::MatrixXf& input)
{
	Eigen::MatrixXf output = input.unaryExpr([](float x) {return std::max(0.0f, x); });
	return output;
}

Eigen::MatrixXf* Activation_SoftMax_Loss_CategoricalCrossentropy::Forward(const Eigen::MatrixXf& input)
{
	m_Inputs = input;

	//Apply loss softmax and loss function to input
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
	return &m_Output;
}

Eigen::MatrixXf Activation_SoftMax_Loss_CategoricalCrossentropy::Backward(const Eigen::VectorXi& y_true)
{
	//The "Input" from the next layer is really just the ouput of this activation function
	//Because it is always the last function
	int samples = m_Output.rows();

	Eigen::MatrixXf result = m_Output;

	for (int i = 0; i < result.rows(); i++)
	{
		result(i, y_true[i]) -= 1;
	}

	//Normalize the results
	m_dInputs = result / samples;

	return m_dInputs;
}

Eigen::MatrixXf Activation_SoftMax_Loss_CategoricalCrossentropy::Predict(const Eigen::MatrixXf& input)
{
	//Apply loss softmax and loss function to input
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

Eigen::VectorXf Loss_CategoricalCrossentropy::Forward(const Eigen::MatrixXf& y_pred, const Eigen::VectorXi& yTrue)
{
	Eigen::VectorXf sampleLosses(y_pred.rows());
	Eigen::MatrixXf yPred = y_pred; //copy

	//Used to clip the values
	const float EPSILON = 0.0000007f;

	for (int i = 0; i < y_pred.rows(); i++)
	{
		for (int j = 0; j < y_pred.cols(); j++)
		{
			//clip the values so that there is no divide by zero chance later
			if (y_pred(i, j) < EPSILON) yPred(i, j) = EPSILON;
			else if (y_pred(i, j) > 1.0f - EPSILON) yPred(i, j) = 1.0f - EPSILON;
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

float Loss_CategoricalCrossentropy::CalculateLoss(const Eigen::MatrixXf& output, const Eigen::VectorXi& yTrue)
{
	Eigen::VectorXf sampleLosses = Forward(output, yTrue);
	return sampleLosses.mean();
}

Eigen::MatrixXf Activation_Linear::Forward(const Eigen::MatrixXf& input)
{
	m_Inputs = input;
	m_Output = input;
	return m_Output;
}

Eigen::MatrixXf Activation_Linear::Backward(const Eigen::VectorXf& dValues)
{
	m_dInputs = dValues;
	return m_dInputs;
}

Eigen::VectorXf Loss_MSE::Forward(const Eigen::MatrixXf& yPred, const Eigen::VectorXf& yTrue)
{
	Eigen::VectorXf sampleLosses;

	for (int i = 0; i < sampleLosses.rows(); i++)
	{
		Eigen::VectorXf row = yPred.row(i);
		float currentY = yTrue[i];

		row = currentY - row.array();
		row = Eigen::square(row.array());
		sampleLosses[i] = row.mean();
	}
	return sampleLosses;
}

Eigen::MatrixXf Loss_MSE::Backward(const Eigen::VectorXf& dValues, const Eigen::VectorXf& yTrue)
{
	int samples = dValues.rows();
	int outputs = dValues.cols();

	m_dInputs = -2 * (yTrue - dValues) / outputs;
	m_dInputs /= samples;

	return m_dInputs;
}
