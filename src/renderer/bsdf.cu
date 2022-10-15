#include "bsdf.cuh"

namespace nagi {

// reference: https://learnopengl.com/PBR/Theory
__device__ glm::vec3 nagi::microfacetBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& albedo, float metallic, float alpha) {
	float G, D;
	glm::vec3 F;
	glm::vec3 h = halfway(-wi, wo);
	float nh = fmaxf(glm::dot(normal, h), 0.f);
	float a2 = alpha * alpha;
	D = alpha / (nh * nh * (a2 - 1.f) + 1.f);
	D = D * D * INV_PI;

	float k = alpha * 0.5f;
	float G1 = fmaxf(glm::dot(normal, -wi), FLT_EPSILON);
	G1 = G1 / (G1 * (1 - k) + k);
	float G2 = fmaxf(glm::dot(normal, wo), FLT_EPSILON);
	G2 = G2 / (G2 * (1 - k) + k);
	G = G1 * G2;

	float hv = fmaxf(glm::dot(h, -wi), 0.f);
	glm::vec3 f0;
	f0 = glm::mix(glm::vec3{ 0.04 }, albedo, metallic); // x * (1 - level) + y * level
	//f0 = albedo * metallic;
	F = f0 + (1.f - f0) * powf(1.f - hv, 5.f);
	return (1.f - metallic) * (1.f - F) * albedo * INV_PI * fmaxf(glm::dot(wo, normal), 0.f) + 
		D * F * G * 0.25f / (fmaxf(glm::dot(-wi, normal), FLT_EPSILON));
}

__device__ glm::vec3 lambertBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& albedo) {
	float lambert = fmaxf(glm::dot(normal, wo), 0.f);

	return albedo * INV_PI * lambert;
}

}