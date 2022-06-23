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

	Data::Data_Return data = Data::ReadDataFromFile("D:/Dev/MLSandbox/MLSandbox/MLSandbox/src/data.txt", 100, 3);

	for (size_t i = 0; i < 1000000; i++)
	{
		nn.ForwardProp(&data.X, data.y);
		nn.BackwardProp(data.y);
		nn.Optimize(optimizer);
		if (i % 100 == 0)
		{
			std::cout << "Accuracy for i = " << i << ": " << nn.CalculateAccuracy(data.y) << std::endl;
			std::cout << "Loss for i = " << i << ": " << nn.CalculateLoss(data.y) << std::endl;
		}

	}
}
