#include "Linear/Matrix.h"

using namespace ml;

#include <stdexcept>

Matrix::Matrix() : Tensor({1, 1})
{}

Matrix::Matrix(int x, int y) : Tensor({x, y})
{}

Matrix::Matrix(const Tensor& tensor) : Tensor(tensor)
{
	if (tensor.getShape().size() > 2) {
		throw std::invalid_argument("Invalid shape of given tensor for matrix initialization");
	} else if (tensor.getShape().size() == 1) { // if given tensor is 1-dim
		this->reshape(dataSize, 1);
	}
}

Matrix& Matrix::reshape(int x, int y) {
	Tensor::reshape({x, y});
	return *this;
}

Matrix Matrix::transpose() const {
	Matrix res(shape[1], shape[0]);
	for (int i = 0; i < shape[1]; ++i) {
		for (int j = 0; j < shape[0]; ++j) {
			res(i, j) = this->operator()(j, i);
		}
	}
	return res;
}

float& Matrix::operator() (int i, int j) {
	return data.at(i * shape.at(1) + j);
}

float Matrix::operator() (int i, int j) const {
	return data.at(i * shape.at(1) + j);
}

Matrix Matrix::operator*(const Matrix& other) const {
	if (other.getShape().size() && shape[1] != other.getShape()[0]) {
		throw std::invalid_argument("Invalid shape of given matrix for multiplication");
	}
	Matrix res(shape[0], other.getShape()[1]);
	for (int i = 0; i < res.getShape()[0]; ++i) {
		for (int j = 0; j < res.getShape()[1]; ++j) {
			res(i, j) = 0.0f;
			for (int k = 0; k < shape[1]; ++k) {
				res(i, j) += this->operator()(i, k) * other(k, j);
			}
		}
	}
	return res;
}

Matrix& Matrix::operator*=(const Matrix& other) {
	if (other.getShape().size() != 2 && shape[1] != other.getShape()[0]) {
		throw std::invalid_argument("Invalid shape of given matrix for multiplication");
	}
	for (int i = 0; i < shape[0]; ++i) {
		for (int j = 0; j < other.getShape()[1]; ++j) {
			this->operator()(i, j) = 0.0f;
			for (int k = 0; k < shape[1]; ++k) {
				this->operator()(i, j) += this->operator()(i, k) * other(k, j);
			}
		}
	}
	return *this;
}


Matrix Matrix::operator*(float scalar) const {
	return Tensor::operator*(scalar);
}

Matrix& Matrix::operator*=(float scalar) {
	Tensor::operator*=(scalar);
	return *this;
}