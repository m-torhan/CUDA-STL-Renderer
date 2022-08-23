#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GL_GLEXT_PROTOTYPES

#include "GL/glew.h"
#include "GL/glut.h"

#include "cuda_gl_interop.h"

#include <stdio.h>
#include <math.h>
#include <thrust/complex.h>

#include "Utils.cuh"
#include "Triangle.cuh"
#include "Point.cuh"

constexpr int window_size{ 768 };
constexpr int max_fps{ 60 };

GLuint bufferObj;
cudaGraphicsResource *resource;
uchar4 *dev_resource;

int mouse_pos[2]{ 0, 0 };

bool update_required{ false };
bool initial_draw{ true };

uint32_t *triangles_count;
Point *origin;
Point *direction;

//__constant__ Triangle object[max_triangles];
Triangle *dev_object;
uint32_t *dev_triangles_count;
Point *dev_origin;
Point *dev_direction;

__global__ void ray_trace_kernel(uchar4 *pixel, Triangle *triangles, uint32_t *triangles_count, Point *origin, Point *direction) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    Point direction_right = Point(direction->y, direction->x, 0);
    direction_right /= 16 * direction_right.length();
    Point direction_up = direction_right.cross_product(*direction);
    direction_up /= 16 * direction_up.length();

    Point ray_origin = *origin;
    Point ray_direction = *direction + 
        direction_right * (x - window_size / 2) +
        direction_up * (y - window_size / 2);

    ray_direction /= ray_direction.length();

    float min_distance = INFINITY;
    const Triangle *closest_triangle = nullptr;

    HitData hit;
    for (int i{ 0 }; i < (*triangles_count); ++i) {
        hit = triangles[i].Hit(ray_origin, ray_direction);
        if (hit.hit && hit.distance < min_distance) {
            closest_triangle = &triangles[i];
        }
    }

    if (nullptr != closest_triangle) {
        pixel[offset] = OBJECT_COLOR;
        pixel[offset].x *= hit.angle_cos;
        pixel[offset].y *= hit.angle_cos;
        pixel[offset].z *= hit.angle_cos;
    }
    else {
        pixel[offset] = BACKGROUND_COLOR;
    }
}

static void cuda_compute_frame(void) {
    size_t size;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &resource, NULL));

    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&dev_resource, &size, resource));

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    ray_trace_kernel<<<grids, threads>>>(dev_resource, dev_object, dev_triangles_count, dev_origin, dev_direction);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("\r%3.0f fps", 1000 / elapsedTime);

    printf("   %07.3f %07.3f %07.3f ", origin->x, origin->y, origin->z);
    printf("   %07.3f %07.3f %07.3f ", direction->x, direction->y, direction->z);

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &resource, NULL));
}

static void draw_func(void) {
    glDrawPixels(window_size, window_size, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

static void key_func(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        // clean up OpenGL and CUDA

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);

        HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
        exit(0);
    }
}

static void mouse_func(int button, int state, int x, int y) {
    mouse_pos[0] = x;
    mouse_pos[1] = y;
    switch (state) {
    case GLUT_DOWN:
        switch (button) {
        case GLUT_RIGHT_BUTTON:
            break;
        case GLUT_LEFT_BUTTON:
            break;
        }
        break;
    case GLUT_UP:
        break;
    }
}

static void motion_func(int x, int y) {
    mouse_pos[0] = x;
    mouse_pos[1] = y;
}

static void idle_func(void) {
    static double time;
    static double delta_time;

    delta_time = (glutGet(GLUT_ELAPSED_TIME) - time) / 1000.0f;
    time = glutGet(GLUT_ELAPSED_TIME);

    constexpr float radius = 40.0f;

    direction->x = 20.0 * sin(time / 1000);
    direction->y = 20.0 * cos(time / 1000);

    printf(" [%f] ", direction->length());

    origin->x = -radius * sin(time / 1000);
    origin->y = -radius * cos(time / 1000);

    cuda_compute_frame();
    glutPostRedisplay();
}

static void resize_func(int width, int height) {
    glutReshapeWindow(window_size, window_size);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Please provide filename. Example usage:\n"
            "\tSTLRenderer.exe object.stl\n");
        return 1;
    }

    auto temp_object = read_stl_binary(argv[1]);

    cudaDeviceProp prop;
    int dev;

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(window_size, window_size);
    glutCreateWindow("STL Renderer");

    glewInit();
    glGenBuffers(1, &bufferObj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, window_size * window_size * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.pciBusID = 38;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsRegisterFlagsWriteDiscard));

    //HANDLE_ERROR(cudaMemcpyToSymbol(object, temp_object.data(), sizeof(Triangle) * min(max_triangles, temp_object.size())));

    HANDLE_ERROR(cudaMalloc((void**)&dev_object, sizeof(Triangle) * temp_object.size()));

    HANDLE_ERROR(cudaMemcpy(dev_object, temp_object.data(), sizeof(Triangle) * min(max_triangles, temp_object.size()), cudaMemcpyHostToDevice));

    HANDLE_ERROR(cudaHostAlloc((void**)&triangles_count, sizeof(uint32_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&origin, sizeof(Point), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&direction, sizeof(Point), cudaHostAllocWriteCombined | cudaHostAllocMapped));

    *triangles_count = min(max_triangles, temp_object.size());
    printf("Triangles count: %d\n", *triangles_count);

    origin->x = 0;
    origin->y = 0;
    origin->z = 0;

    direction->x = 0;
    direction->y = 0;
    direction->z = 0;

    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_triangles_count, triangles_count, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_origin, origin, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&dev_direction, direction, 0));

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}