#ifndef BSDF_CUH
#define BSDF_CUH

#include "common.cuh"

namespace nagi {

__device__ glm::vec3 microFacetBrdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material);

__device__ glm::vec3 lambertBrdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material);

__device__ glm::vec3 transparentBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material);

}

#endif // !BSDF_CUH
