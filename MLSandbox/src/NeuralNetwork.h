#pragma once
#include <vector>
#include "Layer.h"


class NeuralNetwork
{
public:
	NeuralNetwork();
	void AddLayer(unsigned int size);
	inline Layer& GetLayer(unsigned int index) { return m_Layers[index]; };

private:
	std::vector<Layer> m_Layers;
};

