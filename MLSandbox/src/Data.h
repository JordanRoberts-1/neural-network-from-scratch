#pragma once
#include "Eigen/Eigen"
#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <string>

class Data
{
public:
	struct Data_Return
	{
		Eigen::VectorXf X;
		Eigen::MatrixXf y;
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

	//static Data_Return ReadDataFromFile(const std::string& fileName, int points, int classes)
	//{
	//	std::ifstream infile;
	//	infile.open(fileName.c_str());

	//	Data_Return data;
	//	data.X = Eigen::MatrixXf(points * classes, 2);
	//	data.y = Eigen::VectorXi(points * classes);

	//	float a, b = 0.0f;
	//	int row = 0;
	//	int classCount = 0;
	//	for (std::string line; std::getline(infile, line); )   //read stream line by line
	//	{
	//		std::istringstream in(line);      //make a stream for the line itself

	//		in >> data.X(row, 0) >> data.X(row, 1);                  //and read the first whitespace-separated token
	//		data.y[row] = classCount;
	//		row++;
	//		if (row % points == 0) classCount++;
	//	}

	//	infile.close();
	//	return data;
	//}

	//static Data_Return SpiralData(int points, int classes)
	//{
	//	Data_Return data;
	//	data.X = Eigen::MatrixXf(points * classes, 2);
	//	data.X.setZero();

	//	data.y = Eigen::VectorXi(points * classes);
	//	data.y.setZero();

	//	std::random_device rd{};
	//	std::mt19937 gen{ rd() };
	//	std::normal_distribution<> dist{ 0, 1 };

	//	for (size_t i = 0; i < classes; i++)
	//	{
	//		std::vector<float> r = linspace(0.0, 1.0, points);
	//		std::vector<float> t = linspace(i * 4, (i + 1) * 4, points);
	//		for (auto& value : t)
	//		{
	//			value += dist(gen);
	//		}

	//		std::vector<float> sin(points);
	//		std::vector<float> cos(points);
	//		for (size_t j = 0; j < points; j++)
	//		{
	//			sin[j] = r[j] * std::sin(t[j] * 2.5);
	//			cos[j] = r[j] * std::cos(t[j] * 2.5);
	//			data.X(i * points + j, 0) = cos[j];
	//			data.X(i * points + j, 1) = sin[j];
	//			data.y[i * points + j] = i;
	//		}
	//	}
	//	return data;
	//}

	static Data_Return SineData(const std::string& fileName, int numDimensions)
	{
		std::ifstream infile;
		infile.open(fileName.c_str());

		Data_Return data;
		data.X = Eigen::VectorXf(1000);
		data.y = Eigen::MatrixXf(1000, numDimensions);

		int row = 0;
		for (std::string line; std::getline(infile, line); )   //read stream line by line
		{
			std::istringstream in(line);      //make a stream for the line itself

			float a;
			in >> data.X(row, 0) >> a;                  //and read the first whitespace-separated token

			for (int i = 0; i < numDimensions; i++)
			{
				data.y(row, i) = a;
			}

			row++;
		}

		infile.close();
		return data;
	}
};