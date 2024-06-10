#include <iostream>
#include "cuda_kernel.cuh"

int main()
{
	double a[3], b[3], c[3];
	a[0] = 1.0; a[1] = 2.0; a[2] = 3.0;
	b[0] = 4.0; b[1] = 5.0; b[2] = 6.0;
	kernel(a, b, c, 3);
	std::cout << "C = " << c[0] << ", " << c[1] << ", " << c[2] << '\n';
}
