#include "Functions/ActivFunctions.h"
#include "Functions/LossFunctions.h"
#include "Functions/FilterOperations.h"

#include <cmath>

using namespace ml;

Tensor& ml::ReLU(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float { return std::fmax(0.0f, v); });
}

Tensor& ml::ReLUDeriv(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float { return (v > 0.0f); });
}

Tensor& ml::leakyReLU(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float { return std::fmax(0.01f * v, v); });
}

Tensor& ml::leakyReLUDeriv(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float { return (v > 0.0f) ? 1.0f : 0.01f; });
}

Tensor& ml::sigmoid(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float { return v / (1.0f + std::fabs(v)); });
}

Tensor& ml::sigmoidDeriv(Tensor& tensor) {
	return tensor.applyFunction([](float v) -> float {
		float t = 1.0f + std::fabs(v);
		return 1.0f / (t * t);
	});
}

Tensor ml::MSE(const Tensor& predicted, const Tensor& desired) {
	return (predicted - ml::Tensor(predicted.getShape(), desired.getData())).applyFunction([](float v) -> float { return v * v; });
}

Tensor ml::MSEDeriv(const Tensor& predicted, const Tensor& desired) {
	return (predicted - ml::Tensor(predicted.getShape(), desired.getData())) * 2;
}

Tensor ml::correlate2d(const ml::Tensor& input_, const ml::Tensor& kernel, const std::string& type) {
	Tensor input;
	if (type == "valid") {
		input = input_;
	} else if (type == "full") {
		input.reshape({
			input_.getShape()[0] + 2 * (kernel.getShape()[0] - 1),
			input_.getShape()[1] + 2 * (kernel.getShape()[1] - 1)
		});
		input.setBlock({kernel.getShape()[0] - 1, kernel.getShape()[1] - 1}, input_);
	}

	Tensor res({input.getShape()[0] - kernel.getShape()[0] + 1, input.getShape()[1] - kernel.getShape()[1] + 1});
	for (int i = 0; i < res.getShape()[0]; ++i) {
		for (int j = 0; j < res.getShape()[1]; ++j) {
			res({i, j}) = (input.getBlock({i, j}, kernel.getShape()) * kernel).sum();
		}
	}

	return res;
}

Tensor ml::convolve2d(const ml::Tensor& input, const ml::Tensor& kernel, const std::string& type) {
	return ml::correlate2d(input, kernel.reverse(), type);
}