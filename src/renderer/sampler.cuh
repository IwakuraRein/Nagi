#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include "common.cuh"

#include <thrust/random.h>

#define INV_SQRT_THREE 0.577350269189625764509148780501957456f

namespace nagi {

__device__ __host__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth);

inline __device__ __host__ float fresnel(float cosI, float cosT, float iorI, float iorT) {
    float ti = iorT * cosI;
    float it = iorI * cosT;
    float tt = iorT * cosT;
    float ii = iorI * cosI;
    float r1 = (ti - it) / (ti + it);
    float r2 = (ii - tt) / (ii + tt);
    return 0.5f * (r1 * r1 + r2 * r2);
}

__device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

__device__ __host__ glm::vec3 refractSampler(float ior, const glm::vec3& wi, glm::vec3 n, float rnd1);

__device__ __host__ glm::vec3 reflectSampler(
	float metallic, glm::vec3 albedo, const glm::vec3& wi, glm::vec3 n, float rnd1, float rnd2, float rnd3, float* pdf, bool* specular);

__device__ __host__ glm::vec3 ggxImportanceSampler(
	float alpha, const glm::vec3& wi, const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

__device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2);

}

#endif // !SAMPLER_CUH
