#ifndef LAYER_H_
#define LAYER_H_

#include "Linear/Tensor.h"
#include <vector>

namespace ml {
	class Layer
	{
	public:
		virtual ml::Tensor forward(const ml::Tensor& inputs) = 0;
		virtual ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) = 0;
	};
}

#endif