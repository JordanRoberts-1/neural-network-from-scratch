#pragma once
#include <vector>
#include "Layer.h"


class NeuralNetwork
{
public:
	NeuralNetwork();
	void AddLayer(unsigned int numInputs, unsigned int size);
	inline Layer& GetLayer(unsigned int index) { return m_Layers[index]; };

	void ForwardProp(Eigen::MatrixXf input);

private:
	std::vector<Layer> m_Layers;
};

