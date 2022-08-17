#pragma once
#include "cuda_runtime.h"

#include "Point.h"

class Triangle
{
public:
	__host__ __device__ constexpr Triangle() {};
	__host__ __device__ Triangle(Point _a, Point _b, Point _c, Point _normal) : a(_a), b(_b), c(_c), normal(_normal) {};

	__host__ __device__ uchar4 Hit(Point ray_begin, Point ray_direction) const;

private:
	Point a;
	Point b;
	Point c;
	Point normal;
};

