#include "Layers/Convolutional.h"

#include "Functions/FilterOperations.h"

using namespace ml;

Convolutional::Convolutional(const std::vector<int>& inputShape, int depth, const std::vector<int>& kernelShape)
  :	kernelShape(kernelShape), selfDepth(depth), inputDepth(inputShape[0])
{
	std::vector<int> outputShape = {
		selfDepth,
		inputShape[1] - kernelShape[0] + 1,
		inputShape[2] - kernelShape[1] + 1,
	};

	input.reshape(inputShape);
	output.reshape(outputShape);
	biases.reshape(outputShape);
	kernels.reshape({
		selfDepth,
		inputDepth,
		kernelShape[0],
		kernelShape[1],
	});

	kernels.setRandom();
	biases.setRandom();
}

ml::Tensor Convolutional::forward(const ml::Tensor& input) {
	this->input = input;
	output = biases;

	for (int i = 0; i < selfDepth; ++i) {
		ml::Tensor slice = output.slice(i), kernel = kernels.slice(i);
		for (int j = 0; j < inputDepth; ++j) {
			slice += ml::correlate2d(input.slice(j), kernel.slice(j), "valid");
		}
		output.setSlice(i, slice);
	}

	return output;
}

ml::Tensor Convolutional::backward(const ml::Tensor& outputGrad, float learningRate) {
	ml::Tensor kernelsGrad(kernels.getShape());
	ml::Tensor inputGrad(input.getShape());

	for (int i = 0; i < selfDepth; ++i) {
		for (int j = 0; j < inputDepth; ++j) {
			kernelsGrad.setSlice({i, j}, ml::correlate2d(input.slice(j), outputGrad.slice(i), "valid"));
			inputGrad.setSlice(j, inputGrad.slice(j) + ml::convolve2d(outputGrad.slice(i), kernels.slice({i, j}), "full"));
		}
	}

	kernels -= kernelsGrad * learningRate;
	biases -= outputGrad * learningRate;

	return inputGrad;
}