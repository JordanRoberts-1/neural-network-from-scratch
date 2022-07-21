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

void NeuralNetwork::ForwardProp(Eigen::MatrixXf* input)
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
			//Handle the last layer
			Activation_Linear& linear = layer.GetLinear();
			m_CurrentOutput = linear.Forward(*input);

			//std::cout << "Current Output: " << m_CurrentOutput << std::endl;
			//std::cout << "Current Output rows: " << m_CurrentOutput.rows() << std::endl;
		}
	}
}

void NeuralNetwork::BackwardProp(const Eigen::MatrixXf& yTrue)
{
	Eigen::MatrixXf input;

	for (int i = m_Layers.size() - 1; i >= 0; i--)
	{
		Layer& layer = m_Layers[i];

		if (i == m_Layers.size() - 1)
		{
			Loss_MSE& loss = layer.GetMSE();
			input = loss.Backward(layer.GetOutput(), yTrue);

			Activation_Linear& linear = layer.GetLinear();
			input = linear.Backward(input);
		}
		else
		{
			Activation_ReLU& relu = layer.GetReLU();

			input = relu.Backward(input);
		}

		input = layer.Backward(input);
	}
}

void NeuralNetwork::Fit(Eigen::MatrixXf input, const Eigen::MatrixXf& y, const Optimizer_SGD& optimizer)
{
	ForwardProp(&input);
	BackwardProp(y);
	Optimize(optimizer);
}

void NeuralNetwork::Optimize(const Optimizer_SGD& optimizer)
{
	for (auto& layer : m_Layers)
	{
		optimizer.UpdateParams(layer);
	}
}

int NeuralNetwork::Predict(const Eigen::VectorXf& input)
{
	Eigen::MatrixXf result(1, input.size());
	result.row(0) = input;

	//loop through each layer
	for (int i = 0; i < m_Layers.size(); i++)
	{
		Layer& layer = m_Layers[i];
		result = layer.Predict(result);

		if (i != m_Layers.size() - 1)
		{
			//Handle every other layers activation
			Activation_ReLU& activation = layer.GetReLU();
			result = activation.Predict(result);
		}
		else
		{
			//Handle the last layer / loss/ softmax activation
			Activation_SoftMax_Loss_CategoricalCrossentropy& softmax = layer.GetSoftmax();
			result = softmax.Predict(result);
		}
	}

	float max = -1.0f;
	int maxIndex = 0;
	Eigen::VectorXf vector = result.row(0);
	//Find which index is the largest value and return it
	for (int i = 0; i < vector.size(); i++)
	{
		if (vector[i] > max)
		{
			max = vector[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

float NeuralNetwork::CalculateLoss(const Eigen::MatrixXf& yTrue)
{
	return m_Layers[m_Layers.size() - 1].GetMSE().CalculateLoss(m_CurrentOutput, yTrue);
}

float NeuralNetwork::CalculateAccuracy(Eigen::VectorXf yTrue)
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

Eigen::VectorXf NeuralNetwork::GetQs(const Eigen::VectorXf& input)
{
	Eigen::MatrixXf result(1, input.size());
	result.row(0) = input;

	//loop through each layer
	for (int i = 0; i < m_Layers.size(); i++)
	{
		Layer& layer = m_Layers[i];
		result = layer.Predict(result);

		if (i != m_Layers.size() - 1)
		{
			//Handle every other layers activation
			Activation_ReLU& activation = layer.GetReLU();
			result = activation.Predict(result);
		}
		else
		{
			//Handle the last layer / loss/ softmax activation
			Activation_Linear& linear = layer.GetLinear();
			result = linear.Predict(result);
		}
	}

	std::cout << "Result: " << result.row(0) << std::endl;

	return result.row(0);
}

Optimizer_SGD::Optimizer_SGD(float learningRate)
	: m_LearningRate(learningRate)
{
}

void Optimizer_SGD::UpdateParams(Layer& layer) const
{
	layer.UpdateParams(m_LearningRate);
}
