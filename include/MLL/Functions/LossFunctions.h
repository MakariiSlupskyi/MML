#ifndef LOSS_FUNCTIONS_H_
#define LOSS_FUNCTIONS_H_

#include "Linear/Tensor.h"

namespace ml {
	Tensor MSE(const Tensor& predicted, const Tensor& desired);
	Tensor MSEDeriv(const Tensor& predicted, const Tensor& desired);
}

#endif