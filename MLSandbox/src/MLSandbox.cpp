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
	nn.AddLayer(1, 64);
	nn.AddLayer(64, 16);
	nn.AddLayer(16, 2);

	Optimizer_SGD optimizer(0.15f);

	Data::Data_Return data = Data::SineData("C:/Dev/MLSandbox/MLSandbox/MLSandbox/src/sine_data.txt", 2);

	//std::cout << data.y << std::endl;

	Eigen::VectorXf input(1);
	for (int i = 0; i < data.X.size(); i++)
	{
		input[0] = data.X[i];
		std::cout << "X=" << data.X[i] << ",pred=" << nn.GetQs(input) << ", Real Y="
			<< data.y(i, 0) << ", " << data.y(i, 1) << std::endl;
	}

	const int NUM_EPOCHS = 100000;
	for (size_t i = 0; i < NUM_EPOCHS; i++)
	{
		nn.Fit(data.X, data.y, optimizer);
		if (i % 100 == 0)
		{
			//std::cout << "Accuracy for i = " << i << ": " << nn.CalculateAccuracy(data.y) << std::endl;
			std::cout << "Loss for i = " << i << ": " << nn.CalculateLoss(data.y) << std::endl;
		}
	}

	for (int i = 0; i < data.X.size(); i++)
	{
		input[0] = data.X[i];
		std::cout << "X=" << data.X[i] << ",pred=" << nn.GetQs(input) << ", Real Y="
			<< data.y(i, 0) << ", " << data.y(i, 1) << std::endl;
	}
}