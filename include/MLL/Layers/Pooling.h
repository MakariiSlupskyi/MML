#ifndef POOLING_H_
#define POOLING_H_

#include "Layer.h"

#include <vector>
#include "Linear/Tensor.h"

namespace ml {
	class Pooling : public ml::Layer
	{
	public:
		Pooling(int poolSize, int strides);
	
		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) override;

	private:
		int poolSize, strides;
		ml::Tensor input, output;
	};
}

#endif