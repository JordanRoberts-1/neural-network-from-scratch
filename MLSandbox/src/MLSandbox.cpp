#include <iostream>
#include "Eigen/Dense"
#include "NeuralNetwork.h"
#include "Data.h"
#include <iomanip>
#include <random>

int main()
{
	std::cout << "Hello World!\n";

	srand(0);

	NeuralNetwork nn;
	nn.AddLayer(2, 64);
	nn.AddLayer(64, 3);

	Optimizer_SGD optimizer(1.0f);

	Layer& testLayer = nn.GetLayer(0);
	Eigen::MatrixXf testInput(3, 4); //num rows, num Cols
	testInput << 1, 2, 3, 2.5f,
		2.0f, 5.0f, -1.0f, 2.0f,
		-1.5f, 2.7f, 3.3f, -0.8f;

	Eigen::MatrixXf testWeights(3, 4);
	testWeights << 0.2f, 0.8f, -0.5f, 1.0f,
		0.5f, -0.91f, 0.26f, -0.5f,
		-0.26f, -0.27f, 0.17f, 0.87f;
	testWeights.transposeInPlace();

	Eigen::MatrixXf dValues(3, 3);
	dValues << 1.0f, 1.0f, 1.0f,
		2.0f, 2.0f, 2.0f,
		3.0f, 3.0f, 3.0f;

	Eigen::MatrixXf dInputs = dValues * testWeights.transpose();

	std::cout << dInputs << std::endl;

	Eigen::MatrixXf dWeights = testInput.transpose() * dValues;
	std::cout << dWeights << std::endl;

	//Eigen::MatrixXf testZ(3, 4);
	//testZ << 1.0f, 2.0f, -3.0f, -4.0f,
	//	2.0f, -7.0f, -1.0f, 3.0f,
	//	-1.0f, 2.0f, 5.0f, -1.0f;

	//relu activation derivative test
	//Eigen::MatrixXf dRelu = Activation_ReLU::Backward(testZ, dValues);

	Eigen::MatrixXf testBiases(1, 3);

	testBiases = dValues.colwise().sum();

	std::cout << testBiases << std::endl;

	//Eigen::MatrixXf dWeights;

	//dWeights = testInput.transpose() * dValues;
	//std::cout << dWeights << std::endl;

	//Eigen::VectorXf biases(3);
	//biases << 2, 3, 0.5f;

	//std::cout << testInput << std::endl;
	//std::cout << testWeights << std::endl;
	//std::cout << biases << std::endl;

	//testLayer.SetWeightMatrix(testWeights);
	//testLayer.SetBiasVector(biases);





	//Eigen::MatrixXf softmaxOutputs(3, 3);
	//softmaxOutputs << 0.7f, 0.1f, 0.2f,
	//	0.1f, 0.5f, 0.4f,
	//	0.02f, 0.9f, 0.08f;

	//Eigen::VectorXi classTargets(3);
	//classTargets << 0, 1, 1;

	Data::Data_Return data = Data::SpiralData(100, 3);

	for (size_t i = 0; i < 10000; i++)
	{
		nn.ForwardProp(data.X, data.y);
		nn.BackwardProp(data.y);
		nn.Optimize(optimizer);
		if (i % 100 == 0)
		{
			std::cout << "Accuracy for i = " << i << ": " << nn.CalculateAccuracy(data.y) << std::endl;
			std::cout << "Loss for i = " << i << ": " << nn.CalculateLoss(data.y) << std::endl;
		}

	}

	//float data_loss = nn.CalculateLoss(data.y);
	//std::cout << "LOSS: " << data_loss << std::endl;
}
