#include "Linear/Tensor.h"

#include <stdexcept>
#include <cstdlib>
#include <algorithm>

using namespace ml;

// ----------------- DATA GETTERS ----------------- //

Tensor::Tensor() {
	this->reshape({1});
}

Tensor::Tensor(const std::vector<int>& shape) {
	this->reshape(shape);
}

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& values) {
	this->reshape(shape);
	this->setValues(values);
}

Tensor Tensor::slice(int index) const {
	Tensor res(std::vector<int>(shape.begin() + 1, shape.end()));
	std::vector<int> resInds(shape.size() - 1, 0), thisInds(shape.size(), 0);
	
	thisInds[0] = index;
	for (int i = 0; i < res.dataSize; ++i) {
		for (int j = 0; j < resInds.size(); ++j) { thisInds[j + 1] = resInds[j]; }
		res(resInds) = this->operator()(thisInds);
		res.increaseIndices(resInds);
	}
	return res;
}


Tensor Tensor::slice(const std::vector<int>& indices) const {
	if (indices.size() == 0) {
		return *this;
	} else {
		std::vector<int> indices_(indices.begin() + 1, indices.end());
		return this->slice(indices[0]).slice(indices_);
	}
}

Tensor Tensor::getBlock(const std::vector<int>& start, const std::vector<int>& offset) const {
	Tensor res(offset);

	std::vector<int> resInds(shape.size());
	std::vector<int> thisInds(shape.size());

	for (int i = 0; i < res.dataSize; ++i) {
		for (int i = 0; i < offset.size(); ++i) {
			thisInds[i] = resInds[i] + start[i];
		}

		res(resInds) = this->operator()(thisInds);

		res.increaseIndices(resInds);
	}

	return res;
}

Tensor Tensor::reverse() const {
	Tensor res = *this;
	std::reverse(res.data.begin(), res.data.end());
	return res;
}


float Tensor::sum() const {
	float res = 0;
	for (float elem : data) { res += elem; }
	return res;
}

float Tensor::max() const {
	float res = 0;
	for (float elem : data) { 
		if (elem > res) { res = elem; }
	}
	return res;
}

float Tensor::min() const {
	float res = 0;
	for (float elem : data) { 
		if (elem < res) { res = elem; }
	}
	return res;
}

float Tensor::average() const {
	float res = 0;
	for (float elem : data) { res += elem; }
	return res / dataSize;	
}

// ----------------- DATA MODIFIERS ----------------- //


Tensor& Tensor::setValues(const std::vector<float>& values) {
	for (int i = 0; i < dataSize; ++i) {
		data[i] = values.at(i);
	}
	return *this;
}


Tensor& Tensor::setConstant(float values) {
	for (int i = 0; i < dataSize; ++i) {
		data[i] = values;
	}
	return *this;
}

Tensor& Tensor::setRandom() {
	for (int i = 0; i < dataSize; ++i) {
		data[i] = (std::rand() % 100 - 50) / 100.0f;
	}
	return *this;
}

Tensor& Tensor::setSlice(int index, const Tensor& other) {
	if ((other.shape.size() + 1) != shape.size()) {
		throw std::invalid_argument("Invalid shape of given tensor for setSlice() method");
	}
	std::vector<int> otherInds(shape.size() - 1, 0), thisInds(shape.size(), 0);
	
	thisInds[0] = index;
	for (int i = 0; i < other.dataSize; ++i) {
		for (int j = 0; j < otherInds.size(); ++j) { thisInds[j + 1] = otherInds[j]; }
		this->operator()(thisInds) = other(otherInds);
		other.increaseIndices(otherInds);
	}

	return *this;
}

Tensor& Tensor::setSlice(const std::vector<int>& indices, const Tensor& other) {
	if (indices.size() == 0) {
		this->data = other.data;
	} else {
		std::vector<int> indices_(indices.begin() + 1, indices.end());
		this->setSlice(indices[0], this->slice(indices[0]).setSlice(indices_, other));
	}
	return *this;


	//for (int i = 0)
}

Tensor& Tensor::setBlock(const std::vector<int>& start, const Tensor& other) {
	std::vector<int> otherInds(other.shape.size());
	std::vector<int> thisInds(other.shape.size());

	for (int i = 0; i < other.dataSize; ++i) {
		for (int i = 0; i < start.size(); ++i) {
			thisInds[i] = otherInds[i] + start[i];
		}

		this->operator()(thisInds) = other(otherInds);

		other.increaseIndices(otherInds);
	}

	return *this;
}

Tensor& Tensor::reshape(const std::vector<int>& shape) {
	this->shape = shape;
	dataSize = 1;
	for (int i = 0; i < shape.size(); ++i) {
		dataSize *= shape.at(i);
	}
	data.resize(dataSize);
	return *this;
}

Tensor& Tensor::applyFunction(float (*func)(float)) {
	for (int i = 0; i < dataSize; ++i) {
		data[i] = func(data[i]);
	}
	return *this;
}

// ----------------- OPERATORS ----------------- //

float Tensor::operator()(std::vector<int> indices) const {
	return data.at(calcDataIndex(indices));
}

float& Tensor::operator()(std::vector<int> indices) {
	return data.at(calcDataIndex(indices));
}

bool Tensor::operator==(const Tensor& other) const {
	return (shape.size() == other.shape.size() && dataSize == other.dataSize);
}

bool Tensor::operator!=(const Tensor& other) const {
	return !(this->operator==(other));
}

Tensor Tensor::operator+(const Tensor& other) const {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for addition"); }
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] += other.data[i];
	}
	return res;
}

Tensor Tensor::operator-(const Tensor& other) const {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for subtraction"); }
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] -= other.data[i];
	}
	return res;
}

Tensor Tensor::operator*(const Tensor& other) const {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for multilication"); }
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] *= other.data[i];
	}
	return res;
}

Tensor Tensor::operator/(const Tensor& other) const {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for division"); }
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] /= other.data[i];
	}
	return res;
}

Tensor& Tensor::operator+=(const Tensor& other) {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for addition"); }
	for(int i = 0; i < dataSize; ++i) {
		data[i] += other.data[i];
	}
	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for subtraction"); }
	for(int i = 0; i < dataSize; ++i) {
		data[i] -= other.data[i];
	}
	return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for multilication"); }
	for(int i = 0; i < dataSize; ++i) {
		data[i] *= other.data[i];
	}
	return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
	if (*this != other) { throw std::invalid_argument("Invalid shape of given tensor for division"); }
	for(int i = 0; i < dataSize; ++i) {
		data[i] /= other.data[i];
	}
	return *this;
}

Tensor Tensor::operator+(float scalar) const {
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] += scalar;
	}
	return res;
}

Tensor Tensor::operator-(float scalar) const {
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] -= scalar;
	}
	return res;
}

Tensor Tensor::operator*(float scalar) const {
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] *= scalar;
	}
	return res;
}

Tensor Tensor::operator/(float scalar) const {
	Tensor res = *this;
	for(int i = 0; i < dataSize; ++i) {
		res.data[i] /= scalar;
	}
	return res;
}

Tensor& Tensor::operator+=(float scalar) {
	for(int i = 0; i < dataSize; ++i) {
		data[i] += scalar;
	}
	return *this;
}

Tensor& Tensor::operator-=(float scalar) {
	for(int i = 0; i < dataSize; ++i) {
		data[i] -= scalar;
	}
	return *this;
}

Tensor& Tensor::operator*=(float scalar) {
	for(int i = 0; i < dataSize; ++i) {
		data[i] *= scalar;
	}
	return *this;
}

Tensor& Tensor::operator/=(float scalar) {
	for(int i = 0; i < dataSize; ++i) {
		data[i] /= scalar;
	}
	return *this;
}

int Tensor::calcDataIndex(const std::vector<int>& indices) const {
	int index = 0;
	int multiplier = 1;
	for (int i = 0; i < shape.size(); ++i) {
		index += indices[i] * multiplier;
		multiplier *= shape[i];
	}
	return index;
}

std::vector<int>& Tensor::increaseIndices(std::vector<int>& indices) const {
	indices.back() += 1;
	for (int i = shape.size() - 1; i >= 0; --i) {
		if (indices[i] >= shape[i]) {
			indices[i] = 0;
			if (i != 0) { indices[i-1] += 1; }
		} else {
			return indices;
		}
	}
	return indices;
}