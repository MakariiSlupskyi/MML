#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include "Linear/Tensor.h"

namespace ml {
	Tensor& ReLU(Tensor& tensor);
	Tensor& ReLUDeriv(Tensor& tensor);

	Tensor& leakyReLU(Tensor& tensor);
	Tensor& leakyReLUDeriv(Tensor& tensor);

	Tensor& sigmoid(Tensor& tensor);
	Tensor& sigmoidDeriv(Tensor& tensor);

	Tensor& softmax(Tensor& tensor);
	Tensor& softmaxDeriv(Tensor& tensor);

	Tensor MSE(const Tensor& predicted, const Tensor& desired);
	Tensor MSEDeriv(const Tensor& predicted, const Tensor& desired);
}

#endif