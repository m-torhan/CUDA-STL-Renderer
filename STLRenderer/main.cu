#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define GL_GLEXT_PROTOTYPES

#include "GL/glew.h"
#include "GL/glut.h"

#include "cuda_gl_interop.h"

#include <stdio.h>
#include <math.h>
#include <thrust/complex.h>

#include "Utils.h"

constexpr int window_size{ 768 };
constexpr int max_iter{ 1024 };
constexpr float zoom_speed{ 2.0f };

GLuint bufferObj;
cudaGraphicsResource *resource;
uchar4* dev_resource;

int mouse_pos[2]{ 0, 0 };

bool update_required{ false };
bool initial_draw{ true };

__constant__ Triangle object[max_triangles];
uint32_t triangles_count;

Point origin;
Point direction;

__global__ void ray_trace_kernel(uchar4 *pixel, Triangle *triangles, uint32_t triangles_count, Point origin, Point direction) {

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

    ray_trace_kernel<<<grids, threads>>>(dev_resource, object, triangles_count, origin, direction);

    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));

    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("\r%3.0f fps", 1000 / elapsedTime);

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

    HANDLE_ERROR(cudaMemcpyToSymbol(object, temp_object.data(), sizeof(Triangle) * min(max_triangles, temp_object.size())));

    glutKeyboardFunc(key_func);
    glutDisplayFunc(draw_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);
    glutReshapeFunc(resize_func);

    glutMainLoop();
}