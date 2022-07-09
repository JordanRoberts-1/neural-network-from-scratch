#pragma once
#include <Eigen/Eigen>
#include <vector>

class Activation_ReLU
{
public:
	Eigen::MatrixXf* Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::MatrixXf& dValues);

	Eigen::MatrixXf Predict(const Eigen::MatrixXf& input);

	Eigen::MatrixXf GetOutput() const { return m_Output; }
	Eigen::MatrixXf GetdInputs() const { return m_dInputs; }

private:
	Eigen::MatrixXf m_Input;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
};


class Loss_CategoricalCrossentropy
{
public:
	Eigen::VectorXf Forward(const Eigen::MatrixXf& y_pred, const Eigen::VectorXi& yTrue);
	float CalculateLoss(const Eigen::MatrixXf& output, const Eigen::VectorXi& yTrue);

private:
	Eigen::MatrixXf m_dInputs;
};

class Activation_SoftMax_Loss_CategoricalCrossentropy
{
public:
	Eigen::MatrixXf* Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::VectorXi& y_true);

	Eigen::MatrixXf Predict(const Eigen::MatrixXf& input);
	float CalculateLoss(const Eigen::VectorXi& yTrue) { return m_Loss.CalculateLoss(m_Output, yTrue); }

	Eigen::MatrixXf GetOutput() { return m_Output; }
	Eigen::MatrixXf GetdInputs() { return m_dInputs; }

private:
	Loss_CategoricalCrossentropy m_Loss;

	Eigen::MatrixXf m_Inputs;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
};

class Activation_Linear
{
public:
	Eigen::MatrixXf Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::VectorXf& dValues);

	Eigen::MatrixXf Predict(const Eigen::MatrixXf& input);

	Eigen::MatrixXf GetInputs() { return m_Inputs; }
	Eigen::MatrixXf GetOutput() { return m_Output; }
	Eigen::MatrixXf GetdInputs() { return m_dInputs; }

private:
	Eigen::MatrixXf m_Inputs;
	Eigen::MatrixXf m_Output;
	Eigen::MatrixXf m_dInputs;
};

class Loss_MSE
{
public:
	Eigen::VectorXf Forward(const Eigen::MatrixXf& yPred, const Eigen::VectorXf& yTrue);
	Eigen::MatrixXf Backward(const Eigen::VectorXf& dValues, const Eigen::VectorXf& yTrue);
	float CalculateLoss(const Eigen::VectorXf& output, const Eigen::VectorXf& yTrue);

private:
	Eigen::MatrixXf m_dInputs;
};

class Layer
{
public:
	Layer(unsigned int size, unsigned int inputSize);
	Eigen::MatrixXf* Forward(const Eigen::MatrixXf& input);
	Eigen::MatrixXf Backward(const Eigen::MatrixXf& dValues);
	Eigen::MatrixXf Predict(const Eigen::MatrixXf& input);

	inline unsigned int GetSize() const { return m_Size; }
	Activation_ReLU& GetReLU() { return m_ReLU; }
	Activation_Linear& GetLinear() { return m_Linear; }
	Loss_MSE& GetMSE() { return m_MSE; }
	Activation_SoftMax_Loss_CategoricalCrossentropy& GetSoftmax() { return m_Softmax; }

	void UpdateParams(float learningRate);

	Eigen::MatrixXf GetdInputs() { return m_dInputs; }
	Eigen::MatrixXf GetdBiases() { return m_dBiases; }
	Eigen::MatrixXf GetdWeights() { return m_dWeights; }
	Eigen::MatrixXf GetOutput() { return m_Output; }

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

	Activation_ReLU m_ReLU;
	Activation_Linear m_Linear;
	Activation_SoftMax_Loss_CategoricalCrossentropy m_Softmax;
	Loss_MSE m_MSE;

	Eigen::MatrixXf m_WeightMatrix;
	Eigen::VectorXf m_BiasVector;

	Eigen::MatrixXf m_Input;
	Eigen::MatrixXf m_Output;

	Eigen::MatrixXf m_dInputs;
	Eigen::MatrixXf m_dBiases;
	Eigen::MatrixXf m_dWeights;
};

