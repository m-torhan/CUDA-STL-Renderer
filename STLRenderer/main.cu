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
#include "Quaternion.cuh"

constexpr int window_size{ 768 };
constexpr int max_fps{ 60 };

constexpr float rotation_rate{ 2.0f };
constexpr float zoom_rate{ 1.2f };

GLuint buffer_obj;

struct {
    Quaternion rotation{ 1.0f, 0.0f, 0.0f, 0.0f };
    float distance{ 100.0f };
} view;

struct {
    int x;
    int y;
} prev_mouse_pos;

cudaGraphicsResource *opengl_resource;

struct {
    uint32_t *triangles_count;
    Point *origin;
    Quaternion *rotation;
} host_data;

__constant__ Triangle object[max_triangles];
struct {
    uint32_t *triangles_count;
    Point *origin;
    Quaternion *rotation;
    uchar4 *resource;
} device_data;

__global__ void ray_trace_kernel(uchar4 *pixel, uint32_t *triangles_count, Point *origin, Quaternion *rotation) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    Point direction_forward =   rotation->rotate(Point(1, 0, 0));
    Point direction_right =     rotation->rotate(Point(0, 1, 0));
    Point direction_up =        rotation->rotate(Point(0, 0, 1));

    Point ray_origin = *origin;
    Point ray_direction = direction_forward +
        direction_up * (x - window_size / 2) / window_size +
        direction_right * (window_size / 2 - y) / window_size;

    ray_direction /= ray_direction.length();

    const Triangle *closest_triangle = nullptr;
    HitData closest_hit;
    closest_hit.hit = false;

    HitData hit;
    for (int i{ 0 }; i < (*triangles_count); ++i) {
        hit = object[i].Hit(ray_origin, ray_direction);
        if (hit.hit && (!closest_hit.hit || (hit.distance < closest_hit.distance))) {
            closest_triangle = &object[i];
            closest_hit = hit;
        }
    }

    if (nullptr != closest_triangle) {
        pixel[offset] = OBJECT_COLOR;
        pixel[offset].x *= closest_hit.angle_cos;
        pixel[offset].y *= closest_hit.angle_cos;
        pixel[offset].z *= closest_hit.angle_cos;
    }
    else {
        pixel[offset] = BACKGROUND_COLOR;
    }
}

static void cuda_compute_frame(void) {
    size_t size;

    HANDLE_ERROR(cudaGraphicsMapResources(1, &opengl_resource, NULL));

    HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&device_data.resource, &size, opengl_resource));

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));

    HANDLE_ERROR(cudaEventRecord(start, 0));

    dim3 grids(window_size / 16, window_size / 16);
    dim3 threads(16, 16);

    ray_trace_kernel<<<grids, threads>>>(device_data.resource, device_data.triangles_count, device_data.origin, device_data.rotation);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

    printf("\r%3.0f fps ", 1000 / elapsedTime);

    HANDLE_ERROR(cudaGraphicsUnmapResources(1, &opengl_resource, NULL));
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
        glDeleteBuffers(1, &buffer_obj);

        HANDLE_ERROR(cudaGraphicsUnregisterResource(opengl_resource));
        exit(0);
    }
}

static void mouse_func(int button, int state, int x, int y) {
    switch (state) {
    case GLUT_DOWN:
        switch (button) {
        case GLUT_LEFT_BUTTON:
            prev_mouse_pos.x = x;
            prev_mouse_pos.y = y;
            break;
        case 3:
            view.distance /= zoom_rate;
            break;
        case 4:
            view.distance *= zoom_rate;
            break;
        }
        break;
    case GLUT_UP:
        break;
    }
}

static void motion_func(int x, int y) {
    float delta_x = x - prev_mouse_pos.x;
    float delta_y = y - prev_mouse_pos.y;

    Point direction_up = Point(0, 0, 1);
    Point direction_right = Point(0, 1, 0);
       
    view.rotation *= Quaternion(direction_up, rotation_rate * delta_y / window_size)
                   * Quaternion(direction_right, -rotation_rate * delta_x / window_size);

    prev_mouse_pos.x = x;
    prev_mouse_pos.y = y;
}

static void idle_func(void) {
    static double time;
    static double delta_time;

    delta_time = (glutGet(GLUT_ELAPSED_TIME) - time) / 1000.0f;
    time = glutGet(GLUT_ELAPSED_TIME);

    *host_data.rotation = view.rotation;
    *host_data.origin = view.rotation.rotate(Point(-1, 0, 0)) * view.distance;

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
    glGenBuffers(1, &buffer_obj);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer_obj);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, window_size * window_size * 4, NULL, GL_DYNAMIC_DRAW_ARB);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.pciBusID = 38;

    HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

    HANDLE_ERROR(cudaGLSetGLDevice(dev));

    HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&opengl_resource, buffer_obj, cudaGraphicsRegisterFlagsWriteDiscard));

    HANDLE_ERROR(cudaMemcpyToSymbol(object, temp_object.data(), sizeof(Triangle) * min(max_triangles, temp_object.size())));

    HANDLE_ERROR(cudaHostAlloc((void**)&host_data.triangles_count, sizeof(uint32_t), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_data.origin, sizeof(Point), cudaHostAllocWriteCombined | cudaHostAllocMapped));
    HANDLE_ERROR(cudaHostAlloc((void**)&host_data.rotation, sizeof(Quaternion), cudaHostAllocWriteCombined | cudaHostAllocMapped));

    *host_data.triangles_count = min(max_triangles, temp_object.size());
    printf("Triangles count: %d\n", *host_data.triangles_count);

    host_data.origin->x = 0;
    host_data.origin->y = 0;
    host_data.origin->z = 0;

    host_data.rotation->x = 1;
    host_data.rotation->i = 0;
    host_data.rotation->j = 0;
    host_data.rotation->k = 0;

    HANDLE_ERROR(cudaHostGetDevicePointer(&device_data.triangles_count, host_data.triangles_count, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&device_data.origin, host_data.origin, 0));
    HANDLE_ERROR(cudaHostGetDevicePointer(&device_data.rotation, host_data.rotation, 0));

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}