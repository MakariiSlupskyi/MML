#ifndef DENSE_H_
#define DENSE_H_

#include "Layer.h"
#include "Linear/Matrix.h"
#include "Linear/Vector.h"

#include <string>

namespace ml {
	class Dense : public ml::Layer
	{
	public:
		Dense(int inputShape, int outputShape);

		ml::Tensor forward(const ml::Tensor& inputs) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) override;
	
	private:
		Vector input, output, biases;
		Matrix weights;
	};
}

#endif