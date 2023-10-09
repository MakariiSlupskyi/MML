#include "Layers/Flatten.h"

using namespace ml;

#include "Linear/Vector.h"

Flatten::Flatten() {}

ml::Tensor Flatten::forward(const ml::Tensor& input) {
	inputShape = input.getShape();
	return ml::Vector(input.getData());
}

ml::Tensor Flatten::backward(const ml::Tensor& outputGrad, float learningRate) {
	return ml::Tensor(inputShape, outputGrad.getData());
}