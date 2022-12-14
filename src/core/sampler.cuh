#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include "common.cuh"
#include "material.cuh"

#include <thrust/random.h>

namespace nagi {

inline __device__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float phi = rnd * TWO_PI;
	rnd = u01(rng);
	float r = sqrtf(rnd);

	glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };

	pdf = wo_tan.z * INV_PI;

	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	return glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
}

inline __device__ glm::vec3 refractSampler(const float& ior, const glm::vec3& wi, glm::vec3 n, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float invEta, iorI, iorT;
	float cosI = glm::dot(wi, n);
	pdf = 1.f;
	if (cosI < 0) { // enter
		cosI = -cosI;
		n = -n;
		invEta = 1.f / ior;
		iorI = 1.f;
		iorT = ior;
	}
	else { // leave
		n = n;
		invEta = ior;
		iorI = ior;
		iorT = 1.f;
	}

	float cosT = 1.f - invEta * invEta * fmaxf(0.f, (1.f - cosI * cosI));
	if (cosT <= 0.f) { // internal reflection
		return glm::reflect(wi, -n);
	}
	else {
		cosT = sqrtf(cosT);
		float F = fresnel(cosI, cosT, iorI, iorT);

		if (rnd < F) {
			return glm::reflect(wi, -n);
		}
		return wi * invEta + n * (cosT - invEta * cosI);
	}
}

inline __device__ glm::vec3 reflectSampler(
	const float& metallic, const glm::vec3& albedo, const glm::vec3& wi, const glm::vec3& n, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float hv = glm::dot(n, -wi);
	float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	float pdf2;
	glm::vec3 wo;
	if (rnd > F) {
		wo = cosHemisphereSampler(n, pdf2, rng);
	}
	else {
		wo = glm::reflect(wi, n);
		float cosine = fmaxf(glm::dot(wo, n), 0.f);
		pdf2 = sqrtf(1 - cosine * cosine) * INV_PI;
	}

	pdf = pdf2 * (1 - F) + F;
	return wo;
}

// reference: https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
inline __device__ glm::vec3 ggxImportanceSampler(
	const float& alpha, const float& metallic, const glm::vec3 & wi, const glm::vec3 & normal, float& pdf, thrust::default_random_engine& rng){
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);

	float s = 0.5f + metallic / 2.f;
	//float s = metallic;
	float pdf1, pdf2;
	glm::vec3 wo;
	float a2 = alpha * alpha;

	float rnd = u01(rng);
	if (rnd < s) {
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float cosTheta = glm::clamp(sqrtf((1.f - rnd) / (rnd * (a2 - 1.f) + 1)), 0.f, 1.f);
		float cosTheta2 = cosTheta * cosTheta;
		float sinTheta = sqrtf(1.f - cosTheta2);

		glm::vec3 T = getDifferentDir(normal);
		T = glm::normalize(glm::cross(T, normal));
		glm::vec3 B = glm::normalize(glm::cross(T, normal));
		glm::vec3 h{ T * glm::cos(phi) * sinTheta + B * glm::sin(phi) * sinTheta + normal * cosTheta };
		h = glm::normalize(h);
		wo = glm::reflect(wi, h);

		// pdf of ggx
		float denom = ((a2 - 1) * cosTheta2 + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-wi, h), 0.f) + FLT_EPSILON);

		// pdf of cosine hemisphere
		float cosine2 = fmaxf(glm::dot(wo, normal), 0.f);
		pdf2 = sqrtf(1 - cosine2 * cosine2) * INV_PI;
	}
	else {
		wo = cosHemisphereSampler(normal, pdf2, rng);
		glm::vec3 h = halfway(-wi, wo);
		float cosTheta = glm::dot(wo, normal);
		float denom = ((a2 - 1) * cosTheta * cosTheta + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-wi, h), 0.f) + FLT_EPSILON);
	}

	pdf = s * pdf1 + (1.f - s) * pdf2;

	return wo;
}

inline __device__ glm::vec3 uniformHemisphereSampler(const glm::vec3 & normal, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float phi = rnd * TWO_PI;
	rnd = u01(rng);
	float r = sqrtf(1.0f - rnd * rnd);

	glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, rnd };

	pdf = INV_TWO_PI;

	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	return T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z;
}

}

#endif // !SAMPLER_CUH
