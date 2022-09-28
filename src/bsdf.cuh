#ifndef BSDF_CUH
#define BSDF_CUH

#include "common.cuh"

namespace nagi {

__device__ __host__ glm::vec3 opaqueBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const Material& material);

__device__ __host__ glm::vec3 transparentBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const Material& material);

}

#endif // !BSDF_CUH
