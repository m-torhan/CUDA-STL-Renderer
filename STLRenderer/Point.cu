#include "Point.cuh"
#include "Utils.cuh"

const Point& Point::operator=(const Point& other) {
	x = other.x;
	y = other.y;
	z = other.z;

	return *this;
}

const Point Point::operator-() const {
	return Point(-x, -y, -z);
}

Point& Point::operator+=(const Point& other) {
	x += other.x;
	y += other.y;
	z += other.z;

	return *this;
}

const Point Point::operator+(const Point& other) const{
	return Point(x + other.x, y + other.y, z + other.z);
}

Point& Point::operator-=(const Point& other) {
	x -= other.x;
	y -= other.y;
	z -= other.z;

	return *this;
}

const Point Point::operator-(const Point& other) const {
	return Point(x + other.x, y - other.y, z - other.z);
}

Point& Point::operator*=(float number) {
	x *= number;
	y *= number;
	z *= number;

	return *this;
}

const Point Point::operator*(float number) const {
	return Point(x * number, y * number, z * number);
}

const Point operator*(float number, const Point& other) {
	return Point(number * other.x, number * other.y, number * other.z);
}

Point& Point::operator/=(float number) {
	x /= number;
	y /= number;
	z /= number;

	return *this;
}

const Point Point::operator/(float number) const {
	return Point(x / number, y / number, z / number);
}

const Point operator/(float number, const Point& other) {
	return Point(number / other.x, number / other.y, number / other.z);
}

float Point::length() const {
	return sqrtf(x * x + y * y + z * z);
}

float Point::dot_product(const Point& other) const {
	return x * other.x + y * other.y + z * other.z;
}

const Point Point::cross_product(const Point& other) const {
	return Point(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}