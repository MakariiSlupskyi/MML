#ifndef MATRIX_H_
#define MATRIX_H_

#include "Tensor.h"

namespace ml {
	class Matrix : public ml::Tensor
	{
	public:
		Matrix();
		Matrix(int x, int y);
		Matrix(const ml::Tensor& tensor);

		Matrix transpose() const;
		Matrix& reshape(int x, int y);

		float operator() (int i, int j) const;
		float& operator() (int i, int j);

		Matrix operator*(const Matrix& other) const;
		Matrix& operator*=(const Matrix& other);

		Matrix operator*(float scalar) const;
		Matrix& operator*=(float scalar);
	};
}

#endif