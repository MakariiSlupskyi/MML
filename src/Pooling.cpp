#include "Layers/Pooling.h"

#include <vector>
#include <iostream>

using namespace ml;

Pooling::Pooling(int poolSize, int strides) : poolSize(poolSize), strides(strides)
{}

ml::Tensor Pooling::forward(const ml::Tensor& input) {
	this->input = input;
	
	output.reshape({input.getShape()[0], input.getShape()[1] / strides, input.getShape()[2] / strides});

	for (int i = 0; i < output.getShape()[0]; ++i) {
		for (int j = 0; j < output.getShape()[1]; ++j) {
			for (int k = 0; k < output.getShape()[2]; ++k) {
				output({i, j, k}) = input.getBlock(
					{i, j * strides, k * strides},
					{input.getShape()[0], poolSize, poolSize}
				).max();
			}
		}
	}
	return output;
}

ml::Tensor Pooling::backward(const ml::Tensor& outputGrad, float learningRate) {
	ml::Tensor inputGrad(input.getShape());
	for (int i = 0; i < output.getShape()[0]; ++i) {
		for (int j = 0; j < output.getShape()[1]; ++j) {
			for (int k = 0; k < output.getShape()[2]; ++k) {
				inputGrad({i, j * strides, k * strides}) = outputGrad({i, j, k});
			}
		}
	}
	return inputGrad;
}