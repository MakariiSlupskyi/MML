#ifndef VECTOR_H_
#define VECTOR_H_

#include "Matrix.h"

namespace ml {
	class Vector : public ml::Matrix
	{
	public:
		Vector();
		Vector(int size);
		Vector(const std::vector<float>& values);
		Vector(const ml::Tensor& tensor);
	
		Vector& reshape(int size);

		float operator()(int index) const;
		float& operator()(int index);

		ml::Matrix operator*(const ml::Matrix& other) const;
		ml::Matrix& operator*=(const ml::Matrix& other);
	};
}

#endif