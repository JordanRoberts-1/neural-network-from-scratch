#pragma once
#include <Eigen/Eigen>
#include <vector>

class Activation_ReLU
{
public:
	Eigen::MatrixXf Forward(Eigen::MatrixXf input);
	Eigen::MatrixXf Backward(Eigen::MatrixXf dValues);

	//!change this back to private
public:
	Eigen::MatrixXf m_Input;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
};


class Loss_CategoricalCrossentropy
{
public:
	Eigen::VectorXf Forward(Eigen::MatrixXf y_pred, Eigen::VectorXi yTrue);
	float CalculateLoss(Eigen::MatrixXf output, Eigen::VectorXi yTrue);
	//static float Backward(Eigen::MatrixXf dValues, Eigen::VectorXi yTrue);

private:
	Eigen::MatrixXf m_dInputs;
};

class Activation_SoftMax_Loss_CategoricalCrossentropy
{
public:
	Eigen::MatrixXf Forward(Eigen::MatrixXf input, Eigen::VectorXi yTrue);
	Eigen::MatrixXf Backward(Eigen::MatrixXf output, Eigen::VectorXi y_true);
	float CalculateLoss(Eigen::VectorXi yTrue) { return m_Loss.CalculateLoss(m_Output, yTrue); }

	//!change back to private
public:
	Loss_CategoricalCrossentropy m_Loss;

	Eigen::MatrixXf m_Inputs;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
};

class Layer
{
public:
	Layer(unsigned int size, unsigned int inputSize);
	Eigen::MatrixXf Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::MatrixXf& dValues);

	inline unsigned int GetSize() const { return m_Size; }
	Activation_ReLU& GetReLU() { return m_ReLU; }
	Activation_SoftMax_Loss_CategoricalCrossentropy& GetSoftmax() { return m_Softmax; }
	void UpdateParams(float learningRate);

	Eigen::MatrixXf GetdInputs() { return m_dInputs; }
	Eigen::MatrixXf GetdBiases() { return m_dBiases; }
	Eigen::MatrixXf GetdWeights() { return m_dWeights; }

	//Used for debugging
	void SetWeightMatrix(Eigen::MatrixXf input)
	{
		m_WeightMatrix = input;
	}

	void SetBiasVector(Eigen::VectorXf input)
	{
		m_BiasVector = input;
	}

	Eigen::MatrixXf GetOutput() { return m_Output; }

	//!Change this back to private
public:
	unsigned int m_Size;

	Activation_ReLU m_ReLU;
	Activation_SoftMax_Loss_CategoricalCrossentropy m_Softmax;

	Eigen::MatrixXf m_WeightMatrix;
	Eigen::VectorXf m_BiasVector;

	Eigen::MatrixXf m_Input;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
	Eigen::MatrixXf m_dBiases;
	Eigen::MatrixXf m_dWeights;
};

