#include "cuda_runtime.h"
#include "cuda_kernel.cuh"
#include "device_launch_parameters.h"

__global__ void vectorAdditionKernel(double* a, double* b, double* c, int array_size) {
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_id < array_size) {
		c[thread_id] = a[thread_id] + b[thread_id];
	}
}

void kernel(double* a, double* b, double* c, int array_size)
{
	double* device_a, * device_b, * device_c;

	// Allocate device memory
	cudaMalloc(reinterpret_cast<void**>(&device_a), array_size * sizeof(double));
	cudaMalloc(reinterpret_cast<void**>(&device_b), array_size * sizeof(double));
	cudaMalloc(reinterpret_cast<void**>(&device_c), array_size * sizeof(double));

	// Copy data to device
	cudaMemcpy(device_a, a, array_size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, array_size * sizeof(double), cudaMemcpyHostToDevice);

	// Calculate block/grid size
	dim3 block_size(512, 1, 1);
	dim3 grid_size(512 / array_size + 1, 1);

	vectorAdditionKernel<< <grid_size, block_size >> > (device_a, device_b, device_c, array_size);
	cudaMemcpy(c, device_c, array_size * sizeof(double), cudaMemcpyDeviceToHost);
}