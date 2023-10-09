#include "Layers/Dense.h"

#include <string>
#include <stdexcept>

using namespace ml;

Dense::Dense(int inputShape, int outputShape)
	: input(inputShape), output(outputShape), biases(outputShape), weights(outputShape, inputShape)
{
	biases.setRandom();
	weights.setRandom();
}

Tensor Dense::forward(const Tensor& input) {
	this->input = input;
	output = weights * input;
	output += biases;
	return output;
}

Tensor Dense::backward(const Tensor& outputGrad, float learningRate) {
	Matrix weightsGrad = Matrix(outputGrad) * input.transpose();
	weights -= weightsGrad * learningRate;
	biases -= outputGrad * learningRate;
	biases.setConstant(0.0);
	return weights.transpose() * outputGrad;
}