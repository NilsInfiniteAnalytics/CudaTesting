#include <cuda_runtime.h>
#include <iostream>
#include "cuda_kernel.cuh"
#include "lbm_constants.cuh"

int main()
{
	int device;
	cudaGetDevice(&device);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);

	std::cout << "Device Name: " << prop.name << std::endl;
	std::cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes" << std::endl;
	std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes" << std::endl;
	std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
	std::cout << "Warp Size: " << prop.warpSize << std::endl;
	std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max Thread Dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "Max Grid Dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;

	int nx = NX;
	int ny = NY;

	size_t size = nx * ny * sizeof(float) * NL;

	if(size > prop.totalGlobalMem) {
		std::cerr << "The size of the distribution functions is too large for the device." << std::endl;
		return 1;
	}

	if(size < prop.totalGlobalMem) {
		std::cout << "The size of the required memory is " << size << " bytes." << std::endl;
		std::cout << "The size of the distribution functions is less than the total global memory by " << prop.totalGlobalMem - size << " bytes." << std::endl;
	}

	float* device_f;
	cudaMalloc(&device_f, size);

	initalize_distribution_functions_kernel_wrapper(device_f, nx, ny);

	float* local_f = static_cast<float*>(malloc(size));
	cudaMemcpy(local_f, device_f, size, cudaMemcpyDeviceToHost);


	free(local_f);
	cudaFree(device_f);
	return 0;
}
