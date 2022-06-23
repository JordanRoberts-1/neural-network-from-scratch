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

void NeuralNetwork::ForwardProp(Eigen::MatrixXf input, Eigen::VectorXi yTrue)
{
	Layer& firstLayer = m_Layers[0];
	firstLayer.Forward(input);
	Activation_ReLU& firstRelu = firstLayer.GetReLU();
	firstRelu.Forward(firstLayer.m_Output);

	Layer& secondLayer = m_Layers[1];
	secondLayer.Forward(firstRelu.m_Output);
	Activation_SoftMax_Loss_CategoricalCrossentropy& softmax = secondLayer.GetSoftmax();
	softmax.Forward(secondLayer.m_Output, yTrue);

	m_CurrentOutput = softmax.m_Output;
}

void NeuralNetwork::BackwardProp(Eigen::VectorXi yTrue)
{
	Layer& lastLayer = m_Layers[1];
	Activation_SoftMax_Loss_CategoricalCrossentropy& softMax = lastLayer.GetSoftmax();
	softMax.Backward(softMax.m_Output, yTrue);
	lastLayer.Backward(softMax.m_dInputs);

	Layer& firstLayer = m_Layers[0];
	Activation_ReLU& relu = firstLayer.GetReLU();
	relu.Backward(lastLayer.m_dInputs);
	firstLayer.Backward(relu.m_dInputs);
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
