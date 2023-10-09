#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "Layer.h"

#include <string>
#include "Functions/ActivFunctions.h"

namespace ml {
	class Activation : public Layer
	{
	public:
		Activation(const std::string& type)
		{
			if (type == "ReLU") {
				activFunc = ml::ReLU;
				activDeriv = ml::ReLUDeriv;
			} if (type == "leaky ReLU") {
				activFunc = ml::leakyReLU;
				activDeriv = ml::leakyReLUDeriv;
			} if (type == "sigmoid") {
				activFunc = ml::sigmoid;
				activDeriv = ml::sigmoidDeriv;
			}
		}

		ml::Tensor forward(const ml::Tensor& inputs) override {
			this->inputs = inputs;
			return activFunc(this->inputs);
		}

		ml::Tensor backward(const ml::Tensor& outputGrad, float learningRate) override {
			return outputGrad * activDeriv(inputs);
		}

	private:
		Tensor inputs;

		Tensor& (*activFunc)(Tensor&);
		Tensor& (*activDeriv)(Tensor&);
	};
}

#endif