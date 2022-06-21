#pragma once
#include <Eigen/Eigen>
#include <vector>

class Activation_ReLU
{
public:
	static Eigen::MatrixXf Forward(Eigen::MatrixXf input);
	static Eigen::MatrixXf Backward(Eigen::MatrixXf z, Eigen::MatrixXf dValues);
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
	static float Backward(Eigen::MatrixXf dValues, Eigen::VectorXi yTrue);
};

class Layer
{
public:
	Layer(unsigned int size, unsigned int inputSize);
	inline unsigned int GetSize() const { return m_Size; }
	Eigen::MatrixXf Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::MatrixXf& dValues);

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

	Eigen::MatrixXf m_Input;
	Eigen::MatrixXf m_dInputs;
};

