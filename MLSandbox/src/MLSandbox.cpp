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
	nn.AddLayer(64, 16);
	nn.AddLayer(16, 3);

	Optimizer_SGD optimizer(0.2f);

	Data::Data_Return data = Data::ReadDataFromFile("D:/Dev/MLSandbox/MLSandbox/MLSandbox/src/data.txt", 100, 3);

	const int NUM_EPOCHS = 50000;
	for (size_t i = 0; i < NUM_EPOCHS; i++)
	{
		nn.Fit(data.X, data.y, optimizer);
		if (i % 100 == 0)
		{
			std::cout << "Accuracy for i = " << i << ": " << nn.CalculateAccuracy(data.y) << std::endl;
			std::cout << "Loss for i = " << i << ": " << nn.CalculateLoss(data.y) << std::endl;
		}
	}
}
