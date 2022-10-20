#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "common.cuh"

#include <thrust/random.h>

#define PDF_EPSILON 0.0001f
#define INV_SQRT_THREE 0.577350269189625764509148780501957456f

namespace nagi {

struct IntersectInfo {
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	glm::vec3 position;
	int mtlIdx;
};

__inline__ __device__ float fresnel(const float& cosI, const float& cosT, const float& iorI, const float& iorT) {
	float ti = iorT * cosI;
	float it = iorI * cosT;
	float tt = iorT * cosT;
	float ii = iorI * cosI;
	float r1 = (ti - it) / (ti + it);
	float r2 = (ii - tt) / (ii + tt);
	return 0.5f * (r1 * r1 + r2 * r2);
}

__inline__ __device__ glm::vec3 getDifferentDir(const glm::vec3& dir) {
	// Find a direction that is not the dir based of whether or not the
	// dir's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 T;
	if (fabsf(dir.x) < INV_SQRT_THREE) {
		T = glm::vec3(1, 0, 0);
	}
	else if (fabsf(dir.y) < INV_SQRT_THREE) {
		T = glm::vec3(0, 1, 0);
	}
	else {
		T = glm::vec3(0, 0, 1);
	}
	return T;
}

__inline__ __device__ thrust::default_random_engine makeSeededRandomEngine(const int& iter, const int& index, const int& depth) {
	int h = hash((1 << 31) | (depth << 22) | iter) ^ hash(index);
	return thrust::default_random_engine(h);
}

__inline__ __device__ thrust::default_random_engine makeSeededRandomEngine(const int& seed) {
	return thrust::default_random_engine(hash(seed));
}

#define getAlbedo(intersect, mtl) \
	glm::vec3 albedo; \
	if (hasBit(mtl.textures, TEXTURE_TYPE_BASE)) { \
		float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersect.uv.x, intersect.uv.y); \
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z }; \
	} \
	else albedo = mtl.albedo; \

#define getNormal(intersect, mtl) \
	glm::vec3 normal; \
	if (hasBit(mtl.textures, TEXTURE_TYPE_NORMAL)) { \
		glm::mat3 TBN = glm::mat3(intersect.tangent, glm::cross(intersect.normal, intersect.tangent), intersect.normal); \
		float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersect.uv.x, intersect.uv.y); \
		glm::vec3 bump{ texVal.x * 2.f - 1.f, texVal.y * 2.f - 1.f, 1.f }; \
		bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f)); \
		normal = glm::normalize(TBN * bump); \
	} \
	else normal = intersect.normal; \

#define getMetallic(intersect, mtl) \
	float metallic; \
	if (hasBit(mtl.textures, TEXTURE_TYPE_BASE)) { \
		metallic = tex2D<float>(mtl.metallicTex.devTexture, intersect.uv.x, intersect.uv.y); \
	} \
	else metallic = mtl.metallic; \

#define getRoughness(intersect, mtl) \
	float roughness; \
	if (hasBit(mtl.textures, TEXTURE_TYPE_ROUGHNESS)) { \
		roughness = tex2D<float>(mtl.roughnessTex.devTexture, intersect.uv.x, intersect.uv.y); \
	} \
	else roughness =  mtl.roughness; \


namespace Lambert {
// brdf (include lambert)
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl);

__device__ const glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf);

__device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl);

// integrated. brdf / pdf
__device__ bool eval(
	const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng);

}

namespace Microfacet {
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl);

 __device__ glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf);

 __device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl);

 __device__ bool eval(
	 const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng);

}

}

#endif // !MATERIAL_CUH


