#pragma once
#include "Eigen/Eigen"
#include <iostream>
#include <random>

class Data
{
public:
	struct Data_Return
	{
		Eigen::MatrixXf X;
		Eigen::VectorXf y;
	};

	static std::vector<float> linspace(float min, float max, int n)
	{
		std::vector<float> result;
		// vector iterator
		int iterator = 0;

		for (int i = 0; i <= n - 2; i++)
		{
			double temp = min + i * (max - min) / (floor((double)n) - 1);
			result.insert(result.begin() + iterator, temp);
			iterator += 1;
		}

		//iterator += 1;

		result.insert(result.begin() + iterator, max);
		return result;
	}

	static Data_Return SpiralData(int points, int classes)
	{
		Data_Return data;
		data.X = Eigen::MatrixXf(points * classes, 2);
		data.X.setZero();

		data.y = Eigen::VectorXf(points * classes);
		data.y.setZero();

		std::random_device rd{};
		std::mt19937 gen{ rd() };
		std::normal_distribution<> dist{ 0, 1 };

		for (size_t i = 0; i < classes; i++)
		{
			std::vector<float> r = linspace(0.0, 1.0, points);
			std::vector<float> t = linspace(i * 4, (i + 1) * 4, points);
			for (auto& value : t)
			{
				value += dist(gen);
			}

			std::vector<float> sin(points);
			std::vector<float> cos(points);
			for (size_t j = 0; j < points; j++)
			{
				sin[j] = r[j] * std::sin(t[j] * 2.5);
				cos[j] = r[j] * std::cos(t[j] * 2.5);
				data.X(i * points + j, 0) = cos[j];
				data.X(i * points + j, 1) = sin[j];
				data.y[i * points + j] = i;
			}
		}
		return data;
	}
};