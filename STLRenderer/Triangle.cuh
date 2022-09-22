#pragma once
#include "cuda_runtime.h"

#include "Point.cuh"

class Triangle;

typedef struct {
	bool hit;
	float angle_cos;
	float distance;
	Point normal;
} HitData;

class Triangle
{
public:
	__host__ __device__ constexpr Triangle() {};
	__host__ __device__ Triangle(Point _a, Point _b, Point _c) : a(_a), b(_b), c(_c) {};

	__host__ __device__ HitData Hit(const Point& ray_begin, const Point& ray_direction) const;

	Point a;
	Point b;
	Point c;
};

