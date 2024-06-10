#ifndef LBM_CONSTANTS_CUH
#define LBM_CONSTANTS_CUH
#include <corecrt_math.h>
#include <crt/host_defines.h>


// Grid and Block dimensions
#define NX 512
#define NY 512
#define NL 9

// Lattice speeds and weight for D2Q9
__constant__ int cxs[NL] = { 0, 1, 0, -1, 0, 1, -1, -1, 1 };
__constant__ int cys[NL] = { 0, 0, 1, 0, -1, 1, 1, -1, -1 };
__constant__ float wi[NL] = { 4.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f };

const float u_lid = 1.0f;
const float reynolds_number = 10000.0f;
const float lattice_sound_speed = 1.0f / sqrtf(3.0f);

#endif
