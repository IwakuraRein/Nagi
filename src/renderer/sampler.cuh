#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include "common.cuh"

#include <thrust/random.h>

#define INV_SQRT_THREE 0.577350269189625764509148780501957456f

namespace nagi {

__device__ __host__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);

__device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

__device__ __host__ glm::vec3 refractionSampler(float ior, const glm::vec3& wi, glm::vec3 n, float rnd);

__device__ __host__ glm::vec3 ggxImportanceSampler(
	float alpha, const glm::vec3& wi, const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

__device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

}

#endif // !SAMPLER_CUH
