#pragma once
#include <Eigen/Eigen>
#include <vector>

class Activation_ReLU
{
public:
	static Eigen::MatrixXf Forward(Eigen::MatrixXf input);
};

class Activation_SoftMax
{
public:
	static Eigen::MatrixXf Forward(Eigen::MatrixXf input);
};

class Loss_CategoricalCrossentropy
{
public:
	static float Forward(Eigen::MatrixXf y_pred, Eigen::VectorXi yTrue);
};

class Layer
{
public:
	Layer(unsigned int size, unsigned int inputSize);
	inline unsigned int GetSize() const { return m_Size; }
	Eigen::MatrixXf ForwardProp(const Eigen::MatrixXf& input) const;

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
	Eigen::MatrixXf m_WeightMatrix;
	Eigen::VectorXf m_BiasVector;
};

