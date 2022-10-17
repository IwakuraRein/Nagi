#ifndef SAMPLER_CUH
#define SAMPLER_CUH

#include "common.cuh"

#include <thrust/random.h>

#define INV_SQRT_THREE 0.577350269189625764509148780501957456f

namespace nagi {

inline __device__ __host__ glm::vec3 getDifferentDir(const glm::vec3& dir) {
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

inline __device__ __host__
thrust::default_random_engine makeSeededRandomEngine(const int& iter, const int& index, const int& depth) {
	int h = hash((1 << 31) | (depth << 22) | iter) ^ hash(index);
	return thrust::default_random_engine(h);
}

inline __device__ __host__ float fresnel(const float& cosI, const float& cosT, const float& iorI, const float& iorT) {
	float ti = iorT * cosI;
	float it = iorI * cosT;
	float tt = iorT * cosT;
	float ii = iorI * cosI;
	float r1 = (ti - it) / (ti + it);
	float r2 = (ii - tt) / (ii + tt);
	return 0.5f * (r1 * r1 + r2 * r2);
}

inline __device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
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

inline __device__ __host__ glm::vec3 refractSampler(const float& ior, const glm::vec3& wi, glm::vec3 n, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float invEta, iorI, iorT;
	float cosI = glm::dot(wi, n);
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

inline __device__ __host__ glm::vec3 reflectSampler(
	const float& metallic, const glm::vec3& albedo, const glm::vec3& wi, const glm::vec3& n, float& pdf, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float hv = glm::dot(n, -wi);
	float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	float pdf2;
	glm::vec3 wo = cosHemisphereSampler(n, pdf2, rng);
	
	if (rnd < F) wo = glm::reflect(wi, n);
	pdf = pdf2 * (1 - F) + F;
	return wo;
	//float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	//if (rnd < F) {
	//	pdf = 1.f / F;
	//	return glm::reflect(wi, n);
	//}
	//else {
	//	auto wo = cosHemisphereSampler(n, pdf, rng);
	//	pdf /= (1 - F);
	//	return wo;
	//}
}

// reference: https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
inline __device__ __host__ glm::vec3 ggxImportanceSampler(
	const float& alpha, const float& metallic, const glm::vec3 & wi, const glm::vec3 & normal, float& pdf, thrust::default_random_engine& rng){
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);

	float a2 = alpha * alpha;
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

	float denom = ((a2 - 1) * cosTheta2 + 1);
	float pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-wi, h), 0.f) + FLT_EPSILON);

	float pdf2;
	glm::vec3 wo = cosHemisphereSampler(normal, pdf2, rng);

	rnd = u01(rng);
	float s = 0.5f + metallic / 2.f;
	//float s = metallic;
	pdf = s * pdf1 + (1.f - s) * pdf2;

	if (rnd < s) wo = glm::reflect(wi, h);

	return wo;
}

inline __device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3 & normal, float& pdf, thrust::default_random_engine& rng) {
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
