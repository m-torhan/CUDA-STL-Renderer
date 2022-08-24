#include "Utils.cuh"

std::vector<Triangle> read_stl_binary(const std::string& filename) {
	std::vector<Triangle> triangles;

	FILE *stl_file = fopen(filename.c_str(), "rb");

	if (NULL == stl_file) {
		printf("Could not open file: %s.\n", filename.c_str());
		return triangles;
	}

	fseek(stl_file, 80, SEEK_SET);

	uint32_t triangles_count;
	fread(&triangles_count, sizeof(uint32_t), 1, stl_file);

	printf("Triangles to be read: %d.\n", triangles_count);

	for (int i{ 0 }; i < triangles_count; ++i) {
		float normal_data[3]; // to be ignored
		float vertex_data[3][3];
		uint16_t attr_byte_count; // to be ignored

		fread(&normal_data, sizeof(float), 3, stl_file);
		fread(&vertex_data, sizeof(float), 9, stl_file);
		fread(&attr_byte_count, sizeof(uint16_t), 1, stl_file);

		triangles.push_back(Triangle(
			Point(vertex_data[0][0], vertex_data[0][1], vertex_data[0][2]),
			Point(vertex_data[1][0], vertex_data[1][1], vertex_data[1][2]),
			Point(vertex_data[2][0], vertex_data[2][1], vertex_data[2][2])));
	}

	// move object to (0, 0, 0)
	Point min_coords = triangles[0].a;
	Point max_coords = triangles[0].a;

	std::vector<Point> all_points;

	for (auto triangle : triangles) {
		all_points.push_back(triangle.a);
		all_points.push_back(triangle.b);
		all_points.push_back(triangle.c);
	}

	for (auto point : all_points) {
		if (point.x < min_coords.x) {
			min_coords.x = point.x;
		}
		if (point.y < min_coords.y) {
			min_coords.y = point.y;
		}
		if (point.z < min_coords.z) {
			min_coords.z = point.z;
		}

		if (point.x > min_coords.x) {
			max_coords.x = point.x;
		}
		if (point.y > min_coords.y) {
			max_coords.y = point.y;
		}
		if (point.z > min_coords.z) {
			max_coords.z = point.z;
		}
	}

	Point mid_coords = (min_coords + max_coords) / 2;

	for (Triangle& triangle : triangles) {
		triangle.a.x -= mid_coords.x;
		triangle.a.y -= mid_coords.y;
		triangle.a.z -= mid_coords.z;

		triangle.b.x -= mid_coords.x;
		triangle.b.y -= mid_coords.y;
		triangle.b.z -= mid_coords.z;

		triangle.c.x -= mid_coords.x;
		triangle.c.y -= mid_coords.y;
		triangle.c.z -= mid_coords.z;
	}

	/*printf("Loaded triangles:\n");

	for (auto triangle : triangles) {
		printf("  (%0.3f, %0.3f, %0.3f), ", triangle.a.x, triangle.a.y, triangle.a.z);
		printf("(%0.3f, %0.3f, %0.3f), ", triangle.b.x, triangle.b.y, triangle.b.z);
		printf("(%0.3f, %0.3f, %0.3f)\n", triangle.c.x, triangle.c.y, triangle.c.z);
	}*/

	fclose(stl_file);

	return triangles;
}