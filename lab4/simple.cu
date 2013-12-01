// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void simple(float *c) 
{
	//c[threadIdx.x] = threadIdx.x;
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

int main()
{
	float *c = new float[N];

	for(int i = 0; i < N; ++i) {
		c[i] = i*2;
	}

	float *cd;
	const int size = N*sizeof(float);
	
	cudaMalloc( (void**)&cd, size );
	cudaMemcpy( cd, c, size, cudaMemcpyHostToDevice ); 


	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	simple<<<dimGrid, dimBlock>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );
	
	for (int i = 0; i < N; i++)
		printf("%.8f ", c[i]);
	printf("\n");
	for(int i = 0; i < N; ++i) {
		printf("%.8f ",sqrt(i*2));
	}
	printf("\n");

	delete[] c;
	printf("done\n");
	return EXIT_SUCCESS;
}
