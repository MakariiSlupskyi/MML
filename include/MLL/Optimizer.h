#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <vector>

namespace ml {
	using Data = std::vector<ml::Tensor>;

	class Model;
	class Layer;
	
	class Optimizer
	{
	public:
		Optimizer(ml::Model *model, std::vector<ml::Layer*>& layers) : model(model), layers(layers)
		{}

		virtual void train(const ml::Data& trainingData, const ml::Data& labels) = 0;		

	protected:
		ml::Model* model;
		std::vector<ml::Layer*>& layers;
	};
}

#endif