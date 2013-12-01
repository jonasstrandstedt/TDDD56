
#include <stdio.h>
#include "readppm.c"
#ifdef __APPLE__
	#include <GLUT/glut.h>
	#include <OpenGL/gl.h>
#else
	#include <GL/glut.h>
#endif

#define FILTER_SIZE 8
#define BLOCK_SIZE 64
#define GRID_SIZE 512/BLOCK_SIZE
#define BLOCK_DIM BLOCK_SIZE+2*FILTER_SIZE
/*
__global__ void filter(unsigned char *image, unsigned char *out, int n, int m)
{
	__shared__ float block[BLOCK_DIM][BLOCK_DIM];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int sumx, sumy, sumz, k, l;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int grid_height = gridDim.y * blockDim.y;

	//get the global index 
	int global_idx = index_y * grid_width + index_x;
	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;

	// copy data, each thread copys a small part

	int TOTAL_THREADS = BLOCK_SIZE*BLOCK_SIZE;
	int TOTAL_CACHE_DATA = BLOCK_DIM*BLOCK_DIM;

	int copy_times = TOTAL_CACHE_DATA / TOTAL_CACHE_DATA;
	for(int q = 0; q < copy_times; ++q) {
		int h = blockIdx.x * (blockDim.x + FILTER_SIZE*2) + threadIdx.x;
		int g = blockIdx.y * (blockDim.y + FILTER_SIZE*2) + threadIdx.y;

		int motsvarande = i*n +j;
		image[motsvarande*3+0]
		image[motsvarande*3+1]
		image[motsvarande*3+2]
	}

	int lastcopy = TOTAL_CACHE_DATA % TOTAL_THREADS;
	if(local_idx < lastcopy) {

	}



	__syncthreads();



// printf is OK under --device-emulation
//	printf("%d %d %d %d\n", i, j, n, m);

	if (j < n && i < m)
	{
		out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
		out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
		out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
	}
	
	int kernel_size = (2*FILTER_SIZE+1)*(2*FILTER_SIZE+1);
	if (i >= filter_size && i < m-filter_size && j >= filter_size && j < n-filter_size)
		{
			// Filter kernel

			sumx=0;sumy=0;sumz=0;
			for(k=-FILTER_SIZE;k<=FILTER_SIZE;k++)
				for(l=-FILTER_SIZE;l<=FILTER_SIZE;l++)
				{
					sumx += image[((i+k)*n+(j+l))*3+0];
					sumy += image[((i+k)*n+(j+l))*3+1];
					sumz += image[((i+k)*n+(j+l))*3+2];
				}
			out[(i*n+j)*3+0] = sumx/kernel_size;
			out[(i*n+j)*3+1] = sumy/kernel_size;
			out[(i*n+j)*3+2] = sumz/kernel_size;
		}
}
*/


__device__ int element(int x, int y, int width)
{
  return (y*width+x)*3;
}

__global__ void filter_x(unsigned char *image, unsigned char *out, int n, int m) {

	__shared__ unsigned char block[BLOCK_DIM][3];

	int sumx, sumy, sumz, l;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int grid_height = gridDim.y * blockDim.y;

	//get the global index 
	int global_idx = index_y * grid_width + index_x;
	int local_idx = threadIdx.y * blockDim.x + threadIdx.x;



	block[threadIdx.x+FILTER_SIZE][0] = image[element(x, y, n)+0];
	block[threadIdx.x+FILTER_SIZE][1] = image[element(x, y, n)+1];
	block[threadIdx.x+FILTER_SIZE][2] = image[element(x, y, n)+2];

	// IF FIRST
	
	if(threadIdx.x < FILTER_SIZE) {
		int offset = threadIdx.x-FILTER_SIZE;
		block[threadIdx.x][0] = image[element(x+offset, y, n)+0];
		block[threadIdx.x][1] = image[element(x+offset, y, n)+1];
		block[threadIdx.x][2] = image[element(x+offset, y, n)+2];
		//block[threadIdx.x][threadIdx.y][0] = 255;
		//block[threadIdx.x][threadIdx.y][1] = 0;
		//block[threadIdx.x][threadIdx.y][2] = 0;
	}
	
	// IF LAST Threads, copy additional data into 
	if(threadIdx.x > BLOCK_SIZE-FILTER_SIZE-1) {
		int offset = threadIdx.x -BLOCK_SIZE +  FILTER_SIZE+1;
		block[threadIdx.x + 2*FILTER_SIZE][0] = image[element(x+offset, y, n)+0];
		block[threadIdx.x + 2*FILTER_SIZE][1] = image[element(x+offset, y, n)+1];
		block[threadIdx.x + 2*FILTER_SIZE][2] = image[element(x+offset, y, n)+2];
		//block[threadIdx.x + 2*FILTER_SIZE][threadIdx.y][0] = 0;
		//block[threadIdx.x + 2*FILTER_SIZE][threadIdx.y][1] = 0;
		//block[threadIdx.x + 2*FILTER_SIZE][threadIdx.y][2] = 255;
	}

	__syncthreads();

	if (x < n && y < m)
	{
		out[element(x, y, n)+0] = block[threadIdx.x+FILTER_SIZE][0];
		out[element(x, y, n)+1] = block[threadIdx.x+FILTER_SIZE][1];
		out[element(x, y, n)+2] = block[threadIdx.x+FILTER_SIZE][2];
	}
	
	
	int kernel_size = (2*FILTER_SIZE+1);
	if (x >= FILTER_SIZE && x < n-FILTER_SIZE)
	{
		// Filter kernel

		sumx=0;sumy=0;sumz=0;
		for(l=-FILTER_SIZE;l<=FILTER_SIZE;l++)
		{
			int offset = FILTER_SIZE+l;
			offset= 0;
			offset = l;
			//sumx += block[threadIdx.x+offset][0];
			//sumy += block[threadIdx.x+offset][1];
			//sumz += block[threadIdx.x+offset][2];
			sumx += image[element(x+offset, y, n)+0];
			sumy += image[element(x+offset, y, n)+1];
			sumz += image[element(x+offset, y, n)+2];
		}
		out[element(x, y, n)+0] = sumx/kernel_size;
		out[element(x, y, n)+1] = sumy/kernel_size;
		out[element(x, y, n)+2] = sumz/kernel_size;
	}
	

	if (x == 500)
	{
		out[element(x, y, n)+0] = 255;
		out[element(x, y, n)+1] = 0;
		out[element(x, y, n)+2] = 0;
	}
	if (y == 50)
	{
		out[element(x, y, n)+0] = 0;
		out[element(x, y, n)+1] = 255;
		out[element(x, y, n)+2] = 0;
	}

	if (x == y)
	{
		out[element(x, y, n)+0] = 0;
		out[element(x, y, n)+1] = 0;
		out[element(x, y, n)+2] = 255;
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
	
	dim3 dimBlock( BLOCK_SIZE, 1 );
	dim3 dimGrid( GRID_SIZE, GRID_SIZE*BLOCK_SIZE );
	
	filter_x<<<dimGrid, dimBlock>>>(dev_image, dev_out, n, m);
	cudaThreadSynchronize();
	//filter_y<<<dimGrid, dimBlock>>>(dev_out, dev_image, n, m);
	//cudaThreadSynchronize();
	
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
