#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
	: m_Layers()
{
}

void NeuralNetwork::AddLayer(unsigned int size)
{
	if (m_Layers.size() == 0)
	{
		m_Layers.emplace_back(size, 0, true);
		return;
	}
	m_Layers.emplace_back(size, m_Layers.back().GetSize(), false);
}
