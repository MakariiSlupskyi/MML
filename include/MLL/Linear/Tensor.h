#ifndef TENSOR_H_
#define TENSOR_H_

#include <vector>

namespace ml {
	class Tensor
	{
	public:
		Tensor();
		Tensor(const std::vector<int>& shape);
		Tensor(const std::vector<int>& shape, const std::vector<float>& values);

		// Data getters

		std::vector<int> getShape() const { return shape; }
		std::vector<float> getData() const { return data; }
		Tensor slice(int index) const;
		Tensor slice(const std::vector<int>& indices) const;
		Tensor getBlock(const std::vector<int>& start, const std::vector<int>& offset) const;

		Tensor reverse() const;
		float sum() const;
		float max() const;
		float min() const;
		float average() const;

		// Data modifiers

		Tensor& setValues(const std::vector<float>& values);
		Tensor& setConstant(float values);
		Tensor& setRandom();
		Tensor& setSlice(int index, const Tensor& other);
		Tensor& setSlice(const std::vector<int>& indices, const Tensor& other);
		Tensor& setBlock(const std::vector<int>& start, const Tensor& other);

		Tensor& reshape(const std::vector<int>& sizes);
		Tensor& applyFunction(float (*func)(float));

		// Operators

		float operator()(std::vector<int> indices) const;
		float& operator()(std::vector<int> indices);

		bool operator==(const Tensor& other) const;
		bool operator!=(const Tensor& other) const;

		Tensor operator+(const Tensor& other) const;
		Tensor operator-(const Tensor& other) const;
		Tensor operator*(const Tensor& other) const;
		Tensor operator/(const Tensor& other) const;

		Tensor& operator+=(const Tensor& other);
		Tensor& operator-=(const Tensor& other);
		Tensor& operator*=(const Tensor& other);
		Tensor& operator/=(const Tensor& other);

		Tensor operator+(float scalar) const;
		Tensor operator-(float scalar) const;
		Tensor operator*(float scalar) const;
		Tensor operator/(float scalar) const;

		Tensor& operator+=(float scalar);
		Tensor& operator-=(float scalar);
		Tensor& operator*=(float scalar);
		Tensor& operator/=(float scalar);

	protected:
		std::vector<float> data;
		std::vector<int> shape;
		int dataSize;
	
	private:
		int calcDataIndex(const std::vector<int>& indices) const;
		std::vector<int>& increaseIndices(std::vector<int>& indices) const;
	};
}

#endif