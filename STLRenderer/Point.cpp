#include "Point.h"
#include "Utils.h"

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