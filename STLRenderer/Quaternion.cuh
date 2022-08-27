#pragma once

#include <math.h>

#include "Point.cuh"

class Quaternion {
public:
	float x{ 0.0f };
	float i{ 0.0f };
	float j{ 0.0f };
	float k{ 0.0f };

	__host__ __device__ Quaternion() {};
	__host__ __device__ Quaternion(float _x, float _i, float _j, float _k) : x(_x), i(_i), j(_j), k(_k) {};
	__host__ __device__ Quaternion(const Point& axis, float angle);

	__host__ __device__ Quaternion& operator*=(const Quaternion& other);
	__host__ __device__ const Quaternion operator*(const Quaternion& other) const;

	__host__ __device__ const Point rotate(const Point& point) const;
	__host__ __device__ const Quaternion inverse() const;
	__host__ __device__ float length() const;
	__host__ __device__ const Point to_point() const;
};