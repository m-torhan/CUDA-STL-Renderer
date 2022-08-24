#pragma once
#include "cuda_runtime.h"

class Point
{
public:
	float x{ 0.0f };
	float y{ 0.0f };
	float z{ 0.0f };

	__host__ __device__ constexpr Point() = default;
	__host__ __device__ Point(const Point& other) = default;
	__host__ __device__ Point(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {};

	__host__ __device__ const Point& operator=(const Point& other);

	__host__ __device__ ~Point() = default;

	__host__ __device__ const Point operator-() const;

	__host__ __device__ Point& operator+=(const Point& other);
	__host__ __device__ const Point operator+(const Point& other) const;

	__host__ __device__ Point& operator-=(const Point& other);
	__host__ __device__ const Point operator-(const Point& other) const;

	__host__ __device__ Point& operator*=(float number);
	__host__ __device__ const Point operator*(float number) const;
	__host__ __device__ friend const Point operator*(float number, const Point& other);

	__host__ __device__ Point& operator/=(float number);
	__host__ __device__ const Point operator/(float number) const;
	__host__ __device__ friend const Point operator/(float number, const Point& other);

	__host__ __device__ float length() const;

	__host__ __device__ float dot_product(const Point& other) const;
	__host__ __device__ const Point cross_product(const Point& other) const;

	__host__ __device__ void print() const;
};

