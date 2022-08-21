#pragma once
#include <vector>
#include <string>

#include "Triangle.cuh"

constexpr size_t max_triangles = 1024;

#define BACKGROUND_COLOR (uchar4{ 0x1e, 0x1e, 0x1e, 0xff })
#define OBJECT_COLOR (uchar4{0x00, 0x8f, 0xff, 0xff})

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

template<typename T>
__host__ __device__ T min(T a, T b) {
    return a > b ? b : a;
}

std::vector<Triangle> read_stl_binary(const std::string& filename);