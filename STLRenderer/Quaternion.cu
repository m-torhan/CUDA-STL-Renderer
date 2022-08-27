#include "Quaternion.cuh"

Quaternion::Quaternion(const Point& axis, float angle) {
	Point axis_normalized = axis / axis.length();
	angle /= 2;
	x = cosf(angle);
	i = sinf(angle) * axis_normalized.x;
	j = sinf(angle) * axis_normalized.y;
	k = sinf(angle) * axis_normalized.z;
}

Quaternion& Quaternion::operator*=(const Quaternion& other) {
	float tmp[4];
	tmp[0] = x * other.x - i * other.i - j * other.j - k * other.k;
	tmp[1] = x * other.i + i * other.x + j * other.k - k * other.j;
	tmp[2] = x * other.j - i * other.k + j * other.x + k * other.i;
	tmp[3] = x * other.k + i * other.j - j * other.i + k * other.x;
	x = tmp[0];
	i = tmp[1];
	j = tmp[2];
	k = tmp[3];

	return *this;
}

const Quaternion Quaternion::operator*(const Quaternion& other) const {
	Quaternion result = *this;
	result *= other;
	return result;
}

const Point Quaternion::rotate(const Point& point) const {
	Quaternion u = Quaternion(0, point.x, point.y, point.z);
	return ((*this) * u * this->inverse()).to_point();
}

const Quaternion Quaternion::inverse() const {
	return Quaternion(x, -i, -j, -k);
}

float Quaternion::length() const {
	return sqrtf(x * x + i * i + j * j + k * k);
}

const Point Quaternion::to_point() const {
	return Point(i, j, k);
}
