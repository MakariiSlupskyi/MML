#include "Model.h"

#include <stdexcept>

#include "Functions/LossFunctions.h"
#include "Optimizers/SGD.h"

using namespace ml;

Model::Model(const std::vector<Layer*>& layers) : layers(layers), optimizer(nullptr)
{
	if (layers.size() == 0) { throw std::invalid_argument("Model layers wasn't provided"); }
}

void Model::compile(const std::string& optimizerType, const std::string& lossFuncType) {
	lossFunc  = ml::MSE;
	lossDeriv = ml::MSEDeriv;
	
	if (optimizerType == "SGD") {
		optimizer = new SGD(this, this->layers);
	} else {
		throw std::invalid_argument("Invalid optimizer type was provided");
	}
}

Tensor Model::inference(const Tensor& inputData) {
	Tensor output = inputData;
	for (int i = 0; i < layers.size(); ++i) {
		output = layers[i]->forward(output);
	}
	return output;
}

void Model::train(const Data& trainingData, const Data& labels, int epoches) {
	if (optimizer == nullptr) {
		throw std::runtime_error("The model wasn't compiled to use train method");
	}

	for (int i = 0; i < epoches; ++i) {
		optimizer->train(trainingData, labels);
	}
}