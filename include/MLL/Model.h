#ifndef MODEL_H_
#define MODEL_H_

#include <string>
#include <memory>

#include "Layer.h"
#include "Optimizer.h"

namespace ml {
	using Data = std::vector<ml::Tensor>;

	class Model
	{
	public:
		Model(const std::vector<ml::Layer*>& layers);

		void compile(const std::string& optimizerType, const std::string& lossFuncType);

		Tensor inference(const ml::Tensor& inputData);
		void train(const ml::Data& trainingData, const ml::Data& labels, int epoches);

		Tensor (*lossFunc)(const Tensor&, const Tensor&);
		Tensor (*lossDeriv)(const Tensor&, const Tensor&);

	private:
		std::vector<ml::Layer*> layers;
		ml::Optimizer* optimizer;
	};
}

#endif