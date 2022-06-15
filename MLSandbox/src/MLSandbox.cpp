#include <iostream>
#include "Eigen/Dense"
#include "NeuralNetwork.h"

int main()
{
	std::cout << "Hello World!\n";

	NeuralNetwork nn;
	nn.AddLayer(4);
	nn.AddLayer(3);

	Layer& testLayer = nn.GetLayer(1);
	Eigen::MatrixXf testInput(3, 4); //num rows, num Cols
	testInput << 1, 2, 3, 2.5f,
		2.0f, 5.0f, -1.0f, 2.0f,
		-1.5f, 2.7f, 3.3f, -0.8f;

	Eigen::MatrixXf testWeights(3, 4);
	testWeights << 0.2f, 0.8f, -0.5f, 1.0f,
		0.5f, -0.91f, 0.26f, -0.5f,
		-0.26f, -0.27f, 0.17f, 0.87f;

	Eigen::VectorXf biases(3);
	biases << 2, 3, 0.5f;

	std::cout << testInput << std::endl;
	std::cout << testWeights << std::endl;
	std::cout << biases << std::endl;

	testLayer.SetWeightMatrix(testWeights);
	testLayer.SetBiasVector(biases);
	Eigen::MatrixXf result = testLayer.CalculateOutput(testInput);
	std::cout << "Dot product result: " << result << std::endl;
}
