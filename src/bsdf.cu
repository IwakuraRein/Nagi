#include "bsdf.cuh"

namespace nagi {

__device__ __host__ glm::vec3 opaqueBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const Material& material) {
	float lambert = glm::clamp(glm::dot(normal, wo), 0.f, FLT_MAX);
	return material.albedo * INV_PI * lambert;
}

__device__ __host__ glm::vec3 transparentBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const Material& material) {
	float lambert = glm::clamp(glm::dot(normal, wo), 0.f, FLT_MAX);
	return glm::vec3{ 0.f };
}

}