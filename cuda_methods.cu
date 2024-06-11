#include "cuda_runtime.h"
#include "cuda_kernel.cuh"
#include "device_launch_parameters.h"
#include "lbm_constants.cuh"

__global__ void initalize_distribution_functions_kernel(float* f, const int nx, const int ny) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(x < nx && y < ny)
	{
		const int idx = (y * nx + x) * NL;
		for(int k = 0; k < NL; k++)
		{
			f[idx + k] = 1.0;
		}
	}
}

void initalize_distribution_functions_kernel_wrapper(float* device_f, const int nx, const int ny)
{
	dim3 block_size(16, 16);
	dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);
	initalize_distribution_functions_kernel<<<grid_size, block_size>>>(device_f, nx, ny);
	cudaDeviceSynchronize();
}