#pragma once
#include <Eigen/Dense>
class Neuron
{
public:
	Neuron(unsigned int prevLayerCount);
	void Print() const;
	const Eigen::VectorXf& GetWeights() const { return m_Weights; }
	float GetBias() const { return m_Bias; }
private:
	Eigen::VectorXf m_Weights;
	float m_Bias;
};

