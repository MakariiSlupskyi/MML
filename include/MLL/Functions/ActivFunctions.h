#ifndef ACTIV_FUNCTIONS_H_
#define ACTIV_FUNCTIONS_H_

#include "Linear/Tensor.h"

namespace ml {
	Tensor& ReLU(Tensor& tensor);
	Tensor& ReLUDeriv(Tensor& tensor);

	Tensor& leakyReLU(Tensor& tensor);
	Tensor& leakyReLUDeriv(Tensor& tensor);

	Tensor& sigmoid(Tensor& tensor);
	Tensor& sigmoidDeriv(Tensor& tensor);
}

#endif