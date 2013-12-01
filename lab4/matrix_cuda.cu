// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


static struct timeval timeStart;
static char hasStart = 0;

int GetMilliseconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) / 1000 + (tv.tv_sec - timeStart.tv_sec)*1000;
}

int GetMicroseconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) + (tv.tv_sec - timeStart.tv_sec)*1000000;
}

double GetSeconds()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (double)(tv.tv_usec - timeStart.tv_usec) / 1000000.0 + (double)(tv.tv_sec - timeStart.tv_sec);
}

// If you want to start from right now.
void ResetMilli()
{
	struct timeval tv;
	
	gettimeofday(&tv, NULL);
	hasStart = 1;
	timeStart = tv;
}

// If you want to start from a specific time.
void SetMilli(int seconds, int microseconds)
{
	hasStart = 1;
	timeStart.tv_sec = seconds;
	timeStart.tv_usec = microseconds;
}

// problem size (N*N)
const int N = 2048*2; 

// max blocksize
const int blocksize = 16; 

const int grid_N = N / blocksize;

__global__ 
void simple(float *a,float *b,float *c) 
{
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//int index = threadIdx.y * blockDim.x + threadIdx.x;
	
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;
	int grid_height = gridDim.y * blockDim.y;

	//get the global index 
	int global_idx = index_y * grid_width + index_x;
	//int global_idx = index_x * grid_height + index_y;

	c[global_idx] = a[global_idx] + b[global_idx];
}

void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}


int main()
{
	float *a = new float[N*N];
	float *b = new float[N*N];
	float *c = new float[N*N];

	cudaEvent_t myEventStart;
	cudaEvent_t myEventStop;
	cudaEvent_t myEventStart_withoutcopy;
	cudaEvent_t myEventStop_withoutcopy;
	cudaEventCreate(&myEventStart);
	cudaEventCreate(&myEventStop);
	cudaEventCreate(&myEventStart_withoutcopy);
	cudaEventCreate(&myEventStop_withoutcopy);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	}

	float *ad;
	float *bd;
	float *cd;
	const int size = N*N*sizeof(float);
	
	cudaEventRecord(myEventStart, 0);
	cudaEventSynchronize(myEventStart);

	cudaMalloc( (void**)&ad, size );
	cudaMalloc( (void**)&bd, size );
	cudaMalloc( (void**)&cd, size );
	cudaMemcpy( ad, a, size, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, size, cudaMemcpyHostToDevice ); 


	dim3 dimBlock( N / grid_N, N / grid_N );
	dim3 dimGrid( grid_N, grid_N );


	cudaEventRecord(myEventStart_withoutcopy, 0);
	cudaEventSynchronize(myEventStart_withoutcopy);

	simple<<<dimGrid, dimBlock>>>(ad,bd,cd);
	cudaThreadSynchronize();


	cudaEventRecord(myEventStop_withoutcopy, 0);
	cudaEventSynchronize(myEventStop_withoutcopy);
	
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

	cudaFree( ad );
	cudaFree( bd );
	cudaFree( cd );

	cudaEventRecord(myEventStop, 0);
	cudaEventSynchronize(myEventStop);

	float theTime;
	cudaEventElapsedTime(&theTime, myEventStart, myEventStop);


	printf("Problem size (N): %i\n", N);
	printf("blocksize: %i\n", blocksize);
	printf("grid_N: %i\n", grid_N);

	printf("Cuda elapsed time (copy): %f\n", theTime);
	cudaEventElapsedTime(&theTime, myEventStart_withoutcopy, myEventStop_withoutcopy);
	printf("Cuda elapsed time (    ): %f\n", theTime);

	/*
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
	printf("\n");
	*/


	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	}

	int cpu_start = GetMicroseconds();
	
	add_matrix(a, b, c, N);

	int cpu_stop = GetMicroseconds();


	printf("CPU elapsed time: %f\n", (cpu_stop - cpu_start) * 1.0e-3);

	/*
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
	*/
	

	printf("done\n");

	delete[] a;
	delete[] b;
	delete[] c;
	return EXIT_SUCCESS;
}
