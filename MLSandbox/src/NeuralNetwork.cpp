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

void NeuralNetwork::ForwardProp(Eigen::MatrixXf* input, Eigen::VectorXi yTrue)
{
	//loop through each layer
	for (int i = 0; i < m_Layers.size(); i++)
	{
		Layer& layer = m_Layers[i];
		input = layer.Forward(*input);

		if (i != m_Layers.size() - 1)
		{
			//Handle every other layers activation
			Activation_ReLU& activation = layer.GetReLU();
			input = activation.Forward(*input);
		}
		else
		{
			//Handle the last layer / loss/ softmax activation
			Activation_SoftMax_Loss_CategoricalCrossentropy& softmax = layer.GetSoftmax();
			m_CurrentOutput = *softmax.Forward(*input, yTrue);
		}
	}
}

void NeuralNetwork::BackwardProp(Eigen::VectorXi yTrue)
{
	Eigen::MatrixXf input;

	for (int i = m_Layers.size() - 1; i >= 0; i--)
	{
		Layer& layer = m_Layers[i];

		if (i == m_Layers.size() - 1)
		{
			Activation_SoftMax_Loss_CategoricalCrossentropy& softmax = layer.GetSoftmax();
			input = softmax.Backward(yTrue);
		}
		else
		{
			Activation_ReLU& relu = layer.GetReLU();

			input = relu.Backward(input);
		}

		input = layer.Backward(input);
	}
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

	int correctCount = 0;
	for (int i = 0; i < outputChoices.size(); i++)
	{
		if (outputChoices[i] == yTrue[i]) correctCount++;
	}

	float percentCorrect = (float)correctCount / (float)outputChoices.size();
	return percentCorrect;
}

Optimizer_SGD::Optimizer_SGD(float learningRate)
	: m_LearningRate(learningRate)
{
}

void Optimizer_SGD::UpdateParams(Layer& layer)
{
	layer.UpdateParams(m_LearningRate);
}
