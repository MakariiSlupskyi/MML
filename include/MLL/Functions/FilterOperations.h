#ifndef FILTER_OPERATIONS_H
#define FILTER_OPERATIONS_H

#include <string>
#include "Linear/Tensor.h"

namespace ml {
	ml::Tensor correlate2d(const ml::Tensor& input, const ml::Tensor& kenrel, const std::string& type);
	ml::Tensor convolve2d(const ml::Tensor& input, const ml::Tensor& kenrel, const std::string& type);
}

#endif