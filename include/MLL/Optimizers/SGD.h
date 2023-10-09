#ifndef SGD_H_
#define SGD_H_

#include "Optimizer.h"

namespace ml {
	class SGD : public ml::Optimizer
	{
	public:
		SGD(ml::Model* model, std::vector<ml::Layer*>& layers) : Optimizer(model, layers)
		{}

		void train(const ml::Data& trainingData, const ml::Data& labels) override {
			for (int i = 0; i < trainingData.size(); ++i) {
				Tensor error = model->lossDeriv(model->inference(trainingData[i]), labels[i]);
				
				for (int j = layers.size() - 1; j >= 0; --j) {
					error = layers[j]->backward(error, 0.001);
				}
			}
		}
	};
}

#endif