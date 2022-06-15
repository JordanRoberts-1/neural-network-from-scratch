#include "Neuron.h"
#include <iostream>
#include <random>

Neuron::Neuron(unsigned int prevLayerCount)
	: m_Weights()
{
	m_Weights.resize(prevLayerCount);
	m_Weights.setRandom();

	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<double> dist1(-1.0f, 1.0f);

	m_Bias = dist1(rng);
}

void Neuron::Print() const
{
	std::cout << m_Weights << std::endl;
}