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
	nn.AddLayer(2, 5);
	nn.AddLayer(5, 3);

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
	dValues << 1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f,
		7.0f, 8.0f, 9.0f;

	Eigen::MatrixXf testZ(3, 4);
	testZ << 1.0f, 2.0f, -3.0f, -4.0f,
		2.0f, -7.0f, -1.0f, 3.0f,
		-1.0f, 2.0f, 5.0f, -1.0f;

	//relu activation derivative test
	Eigen::MatrixXf dRelu = Activation_ReLU::Backward(testZ, dValues);

	Eigen::MatrixXf testBiases(1, 3);

	for (size_t i = 0; i < testBiases.cols(); i++)
	{
		testBiases(0, i) = dValues.col(i).sum();
	}

	std::cout << testBiases << std::endl;

	Eigen::MatrixXf dWeights;

	dWeights = testInput.transpose() * dValues;
	//std::cout << dWeights << std::endl;


	//Eigen::VectorXf biases(3);
	//biases << 2, 3, 0.5f;

	//std::cout << testInput << std::endl;
	//std::cout << testWeights << std::endl;
	//std::cout << biases << std::endl;

	//testLayer.SetWeightMatrix(testWeights);
	//testLayer.SetBiasVector(biases);

	Data::Data_Return data = Data::SpiralData(100, 3);
	nn.ForwardProp(data.X);

	float data_loss = nn.CalculateLoss(data.y);
	std::cout << "LOSS: " << data_loss << std::endl;
}
