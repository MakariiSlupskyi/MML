#ifndef FLATTEN_H_
#define FLATTEN_H_

#include "Layer.h"

#include <vector>

namespace ml {
	class Flatten : public ml::Layer
	{
	public:
		Flatten();
		
		ml::Tensor forward(const ml::Tensor& input) override;
		ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) override;

	private:
		std::vector<int> inputShape;
	};
}

#endif