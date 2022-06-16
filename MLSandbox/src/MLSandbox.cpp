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

	//Eigen::MatrixXf testWeights(4, 3);
	//testWeights << 0.2f, 0.8f, -0.5f,
	//	1.0f, 0.5f, -0.91f,
	//	0.26f, -0.5f, -0.26f,
	//	-0.27f, 0.17f, 0.87f;

	//Eigen::VectorXf biases(3);
	//biases << 2, 3, 0.5f;

	//std::cout << testInput << std::endl;
	//std::cout << testWeights << std::endl;
	//std::cout << biases << std::endl;

	//testLayer.SetWeightMatrix(testWeights);
	//testLayer.SetBiasVector(biases);

	Data::Data_Return data = Data::SpiralData(100, 3);
	nn.ForwardProp(data.X);
}
