#include "Utils.h"

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
		float normal_data[3];
		float vertex_data[3][3];
		uint16_t attr_byte_count; // to be ignored

		fread(&normal_data, sizeof(float), 3, stl_file);
		fread(&vertex_data, sizeof(float), 9, stl_file);
		fread(&attr_byte_count, sizeof(uint16_t), 1, stl_file);

		triangles.push_back(Triangle(
			Point(vertex_data[0][0], vertex_data[0][1], vertex_data[0][2]),
			Point(vertex_data[1][0], vertex_data[1][1], vertex_data[1][2]),
			Point(vertex_data[2][0], vertex_data[2][1], vertex_data[2][2]),
			Point(normal_data[0], normal_data[1], normal_data[2])));
	}

	fclose(stl_file);

	return triangles;
}