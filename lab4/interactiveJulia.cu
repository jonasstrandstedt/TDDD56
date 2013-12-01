// Ingemars rewrite of the julia demo, integrating the OpenGL parts.
// The CUDA parts are - intentionally - NOT rewritten, and have some
// serious performance problems. Find the problems and make this a¬
// decently performing CUDA program.

// Compile with
// nvcc -lglut -lGL interactiveJulia.cu -o interactiveJulia

#include <GL/glut.h>
#include <GL/gl.h>
#include <stdio.h>
#include "milli.h"


cudaEvent_t myEventStart;
cudaEvent_t myEventStop;
unsigned char *dev_bitmap;

// Image data
	unsigned char	*pixels;
	int	 gImageWidth, gImageHeight;

// Init image data
void initBitmap(int width, int height)
{
	pixels = (unsigned char *)malloc(width * height * 4);
	gImageWidth = width;
	gImageHeight = height;
}

#define DIM 1024

// Complex number class
struct cuComplex
{
    float   r;
    float   i;
    
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    
    __device__ float magnitude2( void )
    {
        return r * r + i * i;
    }
    
    __device__ cuComplex operator*(const cuComplex& a)
    {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    
    __device__ cuComplex operator+(const cuComplex& a)
    {
        return cuComplex(r+a.r, i+a.i);
    }
};

__device__ int julia( int x, int y, float r, float im)
{
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

//    cuComplex c(-0.8, 0.156);
    cuComplex c(r, im);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++)
    {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return i;
    }

    return i;
}

__global__ void kernel( unsigned char *ptr, float r, float im)
{
    // map from blockIdx to pixel position
    //int x = blockIdx.x;
    //int y = blockIdx.y;
    //int offset = x + y * gridDim.x;

    int x = blockIdx.x * blockDim.x + threadIdx.x;    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;

    //get the global index 
    int offset = y * grid_width + x;

    // now calculate the value at that position
    int juliaValue = julia( x, y, r, im );
    ptr[offset*4 + 0] = 255 * juliaValue/200;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

float theReal, theImag;

// Compute CUDA kernel and display image
void Draw()
{


    cudaEventRecord(myEventStart, 0);
    cudaEventSynchronize(myEventStart);
	
    const int blocksize = 16; 
    const int grid_N = DIM / blocksize;

    dim3 dimBlock( blocksize, blocksize );
    dim3 dimGrid( grid_N, grid_N );

	kernel<<<dimGrid, dimBlock>>>( dev_bitmap, theReal, theImag);
	cudaThreadSynchronize();
	cudaMemcpy( pixels, dev_bitmap, gImageWidth*gImageHeight*4, cudaMemcpyDeviceToHost );


    cudaEventRecord(myEventStop, 0);
    cudaEventSynchronize(myEventStop);


    float theTime;
    cudaEventElapsedTime(&theTime, myEventStart, myEventStop);
    printf("Cuda time: %f\n", theTime);
	

    // Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glDrawPixels( gImageWidth, gImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels );
	glutSwapBuffers();
}

void MouseMovedProc(int x, int y)
{
	theReal = -0.5 + (float)(x-400) / 500.0;
	theImag = -0.5 + (float)(y-400) / 500.0;
	//printf("real = %f, imag = %f\n", theReal, theImag);
	glutPostRedisplay ();
}

// Main program, inits
int main( int argc, char** argv) 
{

    cudaEventCreate(&myEventStart);
    cudaEventCreate(&myEventStop);

	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( DIM, DIM );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);
	glutPassiveMotionFunc(MouseMovedProc);
	
	initBitmap(DIM, DIM);

    cudaMalloc( &dev_bitmap, gImageWidth*gImageHeight*4 );
	glutMainLoop();
    cudaFree( dev_bitmap );

    cudaEventDestroy(myEventStart);
    cudaEventDestroy(myEventStop);
}
