#ifndef BSDF_CUH
#define BSDF_CUH

#include "common.cuh"

namespace nagi {

__device__ glm::vec3 specularBrdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& albedo, float metallic);

__device__ glm::vec3 microfacetBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& albedo, float metallic, float roughness);

__device__ glm::vec3 lambertBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& albedo);

}

#endif // !BSDF_CUH
