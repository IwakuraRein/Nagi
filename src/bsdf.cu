#include "bsdf.cuh"

namespace nagi {

__device__ glm::vec3 opaqueBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material) {
	float lambert = glm::clamp(glm::dot(normal, wo), 0.f, FLT_MAX);
	glm::vec3 albedo;
	if (hasTexture(material, TEXTURE_TYPE_BASE)) {
		float4 baseTex = tex2D<float4>(material.baseTex.devTexture, uv.x, uv.y);
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
	}
	else albedo = material.albedo;

	return albedo * INV_PI * lambert;
}

__device__ glm::vec3 transparentBsdf(const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material) {
	float lambert = glm::clamp(glm::dot(normal, wo), 0.f, FLT_MAX);
	return glm::vec3{ 0.f };
}

}