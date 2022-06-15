#pragma once
#include "Neuron.h"
#include <vector>
class Layer
{
public:
	Layer(unsigned int size, unsigned int prevSize, bool isInput);
	inline unsigned int GetSize() const { return m_Size; }
	Eigen::MatrixXf CalculateOutput(const Eigen::MatrixXf& input) const;

	//Used for debugging
	void SetWeightMatrix(Eigen::MatrixXf input)
	{
		m_WeightMatrix = input;
	}

	void SetBiasVector(Eigen::VectorXf input)
	{
		m_BiasVector = input;
	}
private:
	unsigned int m_Size;
	std::vector<Neuron> m_Neurons;
	Eigen::MatrixXf m_WeightMatrix;
	Eigen::VectorXf m_BiasVector;
	bool m_IsInput;
};

