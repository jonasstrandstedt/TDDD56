
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

//#define FILTER_SIZE 2
//#define BLOCK_SIZE 32
//#define GRID_SIZE 512/BLOCK_SIZE
//#define BLOCK_DIM BLOCK_SIZE+2*FILTER_SIZE

#define FILTER_SIZE 2
#define BLOCK_SIZE 16
#define GRID_SIZE 32
#define BLOCK_DIM 20

__device__ int element(int x, int y, int width)
{
  return (y*width+x)*3;
}

__global__ void filter(unsigned char *image, unsigned char *out, int n, int m) {

	__shared__ unsigned char block[BLOCK_DIM][BLOCK_DIM][3];

	int sumx, sumy, sumz, l, k;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int grid_height = gridDim.y * blockDim.y;

	//get the global index 
	int global_idx = index_y * grid_width + index_x;
	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;
	
	int N = BLOCK_DIM;
	int N2 = (BLOCK_DIM)*(BLOCK_DIM);
	int B = BLOCK_SIZE;
	int B2 = (BLOCK_SIZE)*(BLOCK_SIZE);
	
	int block_x0 = blockIdx.x * blockDim.x - FILTER_SIZE;
	int block_y0 = blockIdx.y * blockDim.y - FILTER_SIZE;

	int numReads = ceil( (float)(N2) / (float)(B2) );
	


	for(int i=0; i<numReads; ++i)
	{
		int startOffset = i * B2;
		int threadOffset = startOffset + local_idx;
		
		if(threadOffset >= N2)
			break;
			
		int block_x = threadOffset % N;
		int block_y = threadOffset / N;
		
		int img_x = block_x0 + block_x;
		int img_y = block_y0 + block_y;
		
		block[block_y][block_x][0] = image[element(img_x, img_y, n)+0];
		block[block_y][block_x][1] = image[element(img_x, img_y, n)+1];
		block[block_y][block_x][2] = image[element(img_x, img_y, n)+2];
	}
	
	
/*
	block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][0] = image[element(x, y, n)+0];
	block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][1] = image[element(x, y, n)+1];
	block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][2] = image[element(x, y, n)+2];

	// IF FIRST X
	if(threadIdx.x < FILTER_SIZE) {
		int offset = -FILTER_SIZE;
		block[threadIdx.y+FILTER_SIZE][threadIdx.x][0] = image[element(x+offset, y, n)+0];
		block[threadIdx.y+FILTER_SIZE][threadIdx.x][1] = image[element(x+offset, y, n)+1];
		block[threadIdx.y+FILTER_SIZE][threadIdx.x][2] = image[element(x+offset, y, n)+2];
	}
	
	// IF LAST Threads X
	if(threadIdx.x > BLOCK_SIZE-FILTER_SIZE-1) {
		int offset = FILTER_SIZE;
		block[threadIdx.y+FILTER_SIZE][threadIdx.x + 2*FILTER_SIZE][0] = image[element(x+offset, y, n)+0];
		block[threadIdx.y+FILTER_SIZE][threadIdx.x + 2*FILTER_SIZE][1] = image[element(x+offset, y, n)+1];
		block[threadIdx.y+FILTER_SIZE][threadIdx.x + 2*FILTER_SIZE][2] = image[element(x+offset, y, n)+2];
	}
	
	// IF FIRST Y
	if(threadIdx.y < FILTER_SIZE) {
		int offset = -FILTER_SIZE;
		block[threadIdx.y][threadIdx.x+FILTER_SIZE][0] = image[element(x, y+offset, n)+0];
		block[threadIdx.y][threadIdx.x+FILTER_SIZE][1] = image[element(x, y+offset, n)+1];
		block[threadIdx.y][threadIdx.x+FILTER_SIZE][2] = image[element(x, y+offset, n)+2];
	}
	
	// IF LAST Threads Y
	if(threadIdx.y > BLOCK_SIZE-FILTER_SIZE-1) {
		int offset = FILTER_SIZE;
		block[threadIdx.y+ 2*FILTER_SIZE][threadIdx.x + FILTER_SIZE][0] = image[element(x, y+offset, n)+0];
		block[threadIdx.y+ 2*FILTER_SIZE][threadIdx.x + FILTER_SIZE][1] = image[element(x, y+offset, n)+1];
		block[threadIdx.y+ 2*FILTER_SIZE][threadIdx.x + FILTER_SIZE][2] = image[element(x, y+offset, n)+2];
	}
	
	// UPPER LEFT
	if(threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE)
	{
		int offset = -FILTER_SIZE;
		block[threadIdx.y][threadIdx.x][0] = image[element(x+offset, y+offset, n)+0];
		block[threadIdx.y][threadIdx.x][1] = image[element(x+offset, y+offset, n)+1];
		block[threadIdx.y][threadIdx.x][2] = image[element(x+offset, y+offset, n)+2];
	}
	
	// UPPER RIGHT
	if(threadIdx.x > BLOCK_SIZE-FILTER_SIZE-1 && threadIdx.y < FILTER_SIZE)
	{
		int offsety = -FILTER_SIZE;
		int offsetx = FILTER_SIZE;
		block[threadIdx.y][threadIdx.x+2*FILTER_SIZE][0] = image[element(x+offsetx, y + offsety, n)+0];
		block[threadIdx.y][threadIdx.x+2*FILTER_SIZE][1] = image[element(x+offsetx, y + offsety, n)+1];
		block[threadIdx.y][threadIdx.x+2*FILTER_SIZE][2] = image[element(x+offsetx, y + offsety, n)+2];
	}
	
	// LOWER LEFT
	if(threadIdx.x < FILTER_SIZE && threadIdx.y > BLOCK_SIZE-FILTER_SIZE-1)
	{
		int offsety = FILTER_SIZE;
		int offsetx = -FILTER_SIZE;
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x][0] = image[element(x+offsetx, y+offsety, n)+0];
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x][1] = image[element(x+offsetx, y+offsety, n)+1];
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x][2] = image[element(x+offsetx, y+offsety, n)+2];
	}
	
	// LOWER RIGHT
	if(threadIdx.x > BLOCK_SIZE-FILTER_SIZE-1 && threadIdx.y > BLOCK_SIZE-FILTER_SIZE-1)
	{
		int offset = FILTER_SIZE;
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x+2*FILTER_SIZE][0] = image[element(x+offset, y + offset, n)+0];
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x+2*FILTER_SIZE][1] = image[element(x+offset, y + offset, n)+1];
		block[threadIdx.y+2*FILTER_SIZE][threadIdx.x+2*FILTER_SIZE][2] = image[element(x+offset, y + offset, n)+2];
	}
*/
	__syncthreads();

	if (x < n && y < m)
	{
		out[element(x, y, n)+0] = block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][0];
		out[element(x, y, n)+1] = block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][1];
		out[element(x, y, n)+2] = block[threadIdx.y+FILTER_SIZE][threadIdx.x+FILTER_SIZE][2];
	}
	
	
	int kernel_size = (2*FILTER_SIZE+1)*(2*FILTER_SIZE+1);
	if (x >= FILTER_SIZE && x < n-FILTER_SIZE && y >= FILTER_SIZE && y <m-FILTER_SIZE)
	{
		// Filter kernel

		sumx=0;sumy=0;sumz=0;
		for(k=-FILTER_SIZE;k<=FILTER_SIZE;k++)
			for(l=-FILTER_SIZE;l<=FILTER_SIZE;l++)
			{
				int offset_x = FILTER_SIZE+l;
				int offset_y = FILTER_SIZE+k;
				sumx += block[threadIdx.y+offset_y][threadIdx.x+offset_x][0];
				sumy += block[threadIdx.y+offset_y][threadIdx.x+offset_x][1];
				sumz += block[threadIdx.y+offset_y][threadIdx.x+offset_x][2];
			}
		out[element(x, y, n)+0] = sumx/kernel_size;
		out[element(x, y, n)+1] = sumy/kernel_size;
		out[element(x, y, n)+2] = sumz/kernel_size;
	}
	
	
	
}

__global__ void filter_y(unsigned char *image, unsigned char *out, int n, int m) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int sumx, sumy, sumz, l;

	if (j < n && i < m)
	{
		out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
		out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
		out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
	}
	
	int kernel_size = (2*FILTER_SIZE+1);
	if (i >= FILTER_SIZE && i < m-FILTER_SIZE)
	{
		// Filter kernel

		sumx=0;sumy=0;sumz=0;
		for(l=-FILTER_SIZE;l<=FILTER_SIZE;l++)
		{
			sumx += image[((i+l)*n+(j))*3+0];
			sumy += image[((i+l)*n+(j))*3+1];
			sumz += image[((i+l)*n+(j))*3+2];
		}
		out[(i*n+j)*3+0] = sumx/kernel_size;
		out[(i*n+j)*3+1] = sumy/kernel_size;
		out[(i*n+j)*3+2] = sumz/kernel_size;
	}
}


// Compute CUDA kernel and display image
void Draw()
{
	unsigned char *image, *out;
	int n, m;
	unsigned char *dev_image, *dev_out;
	
	image = readppm("maskros512.ppm", &n, &m);
	out = (unsigned char*) malloc(n*m*3);
	
	cudaMalloc( (void**)&dev_image, n*m*3);
	cudaMalloc( (void**)&dev_out, n*m*3);
	cudaMemcpy( dev_image, image, n*m*3, cudaMemcpyHostToDevice);
	
	dim3 dimBlock( BLOCK_SIZE, BLOCK_SIZE );
	dim3 dimGrid( GRID_SIZE, GRID_SIZE);
	
	filter<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	
	cudaMemcpy( out, dev_out, n*m*3, cudaMemcpyDeviceToHost );
	cudaFree(dev_image);
	cudaFree(dev_out);
	
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glRasterPos2f(-1, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
	glRasterPos2i(0, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( 1024, 512 );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);
	
	glutMainLoop();
}
