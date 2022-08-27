#pragma once

#include <math.h>

#include "Point.cuh"

class Quaternion {
public:
	float x{ 0.0f };
	float i{ 0.0f };
	float j{ 0.0f };
	float k{ 0.0f };

	Quaternion() {};
	Quaternion(float _x, float _i, float _j, float _k) : x(_x), i(_i), j(_j), k(_k) {};
	Quaternion(const Point& axis, float angle);

	Quaternion& operator*=(const Quaternion& other);
	const Quaternion operator*(const Quaternion& other) const;

	const Point rotate(const Point& point) const;
	const Quaternion inverse() const;
	float length() const;
	const Point to_point() const;
};