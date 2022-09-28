#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include "common.cuh"

#include <thrust/random.h>

namespace nagi {

__device__ __host__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);

__device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float* pdf, thrust::default_random_engine& rng);

__device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3& normal, float* pdf, thrust::default_random_engine& rng);

}

#endif // !SAMPLER_CUH
