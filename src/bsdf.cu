#include "bsdf.cuh"

namespace nagi {

// reference: https://learnopengl.com/PBR/Theory
__device__ glm::vec3 nagi::microFacetBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& n, const Material& mtl) {
	float D, G;
	glm::vec3 F;
	glm::vec3 h = halfway(wi, wo);

	glm::vec3 albedo;
	if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
		float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, uv.x, uv.y);
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
	}
	else albedo = mtl.albedo;

	float alpha;
	if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
		alpha = tex2D<float>(mtl.roughnessTex.devTexture, uv.x, uv.y);
	}
	else alpha = mtl.roughness;

	float nh = fmaxf(glm::dot(n, h), 0.f);
	float r2 = alpha * alpha;
	D = alpha / (nh * nh * (r2 - 1.f) + 1.f);
	D = D * D * INV_PI;

	float k = (alpha + 1.f);
	k = k * k * 0.125;
	float G1 = fmaxf(glm::dot(n, wi), 0.f);
	G1 = G1 / (G1 * (1 - k) + k);
	float G2 = fmaxf(glm::dot(n, wo), 0.f);
	G2 = G2 / (G2 * (1 - k) + k);
	G = G1 * G2;

	float hv = fmaxf(glm::dot(h, wi), 0.f);
	glm::vec3 f0;
	float metallic;
	if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
		metallic = tex2D<float>(mtl.metallicTex.devTexture, uv.x, uv.y);
	}
	else metallic = mtl.metallic;
	//f0 = glm::mix(glm::vec3{ 0.04 }, albedo, metallic); // x * (1 - level) + y * level
	f0 = albedo * metallic;
	F = f0 + (1.f - f0) * powf(1.f - hv, 5.f);

	return (1.f - metallic) * (1.f - F) * albedo * INV_PI * fmaxf(glm::dot(wo, n), 0.f) + 
		(1.f - metallic) * D * F * G * 0.25f / (fmaxf(glm::dot(wi, n), 0.f) + FLT_EPSILON);
}

__device__ glm::vec3 lambertBrdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material) {
	float lambert = fmaxf(glm::dot(normal, wo), 0.f);
	glm::vec3 albedo;
	if (hasTexture(material, TEXTURE_TYPE_BASE)) {
		float4 baseTex = tex2D<float4>(material.baseTex.devTexture, uv.x, uv.y);
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
	}
	else albedo = material.albedo;

	return albedo * INV_PI * lambert;
}

__device__ glm::vec3 transparentBsdf(
	const glm::vec3& wi, const glm::vec3& wo, const glm::vec2& uv, const glm::vec3& normal, const Material& material) {
	return glm::vec3{ 0.f };
}

}