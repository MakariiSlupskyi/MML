#include "Linear/Vector.h"

using namespace ml;

#include <stdexcept>

Vector::Vector() : Matrix()
{}

Vector::Vector(int size) : Matrix(size, 1)
{}

Vector::Vector(const std::vector<float>& values) : Matrix(values.size(), 1)
{
	this->setValues(values);
}

Vector::Vector(const ml::Tensor& tensor) : Matrix(tensor)
{
	this->reshape(dataSize);
}

Vector& Vector::reshape(int size) {
	Matrix::reshape(size, 1);
	return *this;
}

float Vector::operator()(int index) const {
	return data.at(index);
}

float& Vector::operator()(int index) {
	return data.at(index);
}

Matrix Vector::operator*(const Matrix& other) const {
	Matrix res = Matrix(*this) * other;
	return res;
}

Matrix& Vector::operator*=(const Matrix& other) {
	Matrix(*this) *= other;
	return *this;
}