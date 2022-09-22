#include "Triangle.cuh"
#include "Utils.cuh"

HitData Triangle::Hit(const Point& ray_begin, const Point& ray_direction) const {
	HitData result{ false, 0, 0, Point() };

	// normal in point a
	Point normal = (b - a).cross_product(c - a);

	if (normal.dot_product(ray_direction) > 0) {
		return result;
	}

	// intersection point of triangle's plane and ray
	Point q = ray_begin + ((a - ray_begin).dot_product(normal) / ray_direction.dot_product(normal)) * ray_direction;

	// check if intersection point is in the inside of the triangle
	Point ab_normal = (b - a).cross_product(normal);
	Point bc_normal = (c - b).cross_product(normal);
	Point ca_normal = (a - c).cross_product(normal);

	int sign_sum = signbit(ab_normal.dot_product(q - a)) + 
				   signbit(bc_normal.dot_product(q - b)) + 
				   signbit(ca_normal.dot_product(q - c));

	// if all signs of dot products are - or + then point q lays on the same side of vectors ab, bc and ca
	if (3 == abs(sign_sum)) {
		result.hit = true;
		result.angle_cos = abs(ray_direction.cosine(normal));
		result.distance = (q - ray_begin).length();
		result.normal = normal;
	}

	return result;
}