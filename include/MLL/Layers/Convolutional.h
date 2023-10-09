#ifndef CONVOLUTIONAL_H_
#define CONVOLUTIONAL_H_

#include "Layer.h"
#include "Linear/Tensor.h"

#include <vector>

namespace ml {
	class Convolutional : public Layer
	{
	public:
		Convolutional(const std::vector<int>& inputShape, int selfDepth, const std::vector<int>& kernelShape = {3, 3});

		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) override;

	private:
		ml::Tensor input, output, biases, kernels;
		std::vector<int> kernelShape;
		int selfDepth, inputDepth;
	};
}

#endif