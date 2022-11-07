#include "material.cuh"

namespace nagi {

namespace Lambert {
// brdf (include lambert)
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getAlbedo(intersect, mtl)
	getNormal(intersect, mtl)
	return albedo * INV_PI * fmaxf(glm::dot(normal, l), 0.f);
}

__device__ const glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf) {
	// cosine hemisphere
	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float phi = rnd * TWO_PI;
	rnd = u01(rng);
	float r = sqrtf(rnd);

	glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };

	pdf = wo_tan.z * INV_PI;

	getNormal(intersect, mtl)
	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	return glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
}

__device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getNormal(intersect, mtl)
	float cosine = fmaxf(glm::dot(l, normal), 0.f);
	return sqrtf(1 - cosine * cosine) * INV_PI;
}

// integrated. brdf / pdf
__device__ bool eval(
	const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng) {
	getNormal(intersect, mtl)
	if (glm::dot(normal, v) >= 0.f) return false;
	getAlbedo(intersect, mtl)

	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float phi = rnd * TWO_PI;
	rnd = u01(rng);
	float r = sqrtf(rnd);

	glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };

	float pdf = wo_tan.z * INV_PI;
	if (pdf < PDF_EPSILON) return false;

	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	l = glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
	float lambert = glm::dot(normal, l);
	if (lambert <= 0.f) return false;

	eval = albedo * INV_PI * lambert / pdf;

	return true;
}

}

namespace Microfacet {
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getNormal(intersect, mtl)
	if (glm::dot(v, normal) >= 0.f) return glm::vec3{ 0.f };
	getAlbedo(intersect, mtl)
	getMetallic(intersect, mtl)
	getRoughness(intersect, mtl)
	float G, D;
	glm::vec3 h = halfway(-v, l);
	float nh = fmaxf(glm::dot(normal, h), 0.f);
	float a2 = roughness * roughness;
	D = roughness / (nh * nh * (a2 - 1.f) + 1.f);
	D = D * D * INV_PI;

	float k = roughness * 0.5f;
	float G1 = fmaxf(glm::dot(normal, -v), FLT_EPSILON);
	G1 = G1 / (G1 * (1 - k) + k);
	float G2 = fmaxf(glm::dot(normal, l), FLT_EPSILON);
	G2 = G2 / (G2 * (1 - k) + k);
	G = G1 * G2;

	float hv = fmaxf(glm::dot(h, -v), 0.f);
	//glm::vec3 F = glm::mix(ior, albedo, metallic); // x * (1 - level) + y * level
	glm::vec3 F = albedo * metallic;
	F = F + (1.f - F) * powf(1.f - hv, 5.f);
	return (1.f - metallic) * (1.f - F) * albedo * INV_PI * fmaxf(glm::dot(l, normal), 0.f) +
		D * F * G * 0.25f / (fmaxf(glm::dot(-v, normal), FLT_EPSILON));
}

__device__ glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	getMetallic(intersect, mtl)
	getRoughness(intersect, mtl)
	getNormal(intersect, mtl)

	float s = 0.5f + metallic / 2.f;
	//float s = metallic;
	float a2 = roughness * roughness;

	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	float rnd = u01(rng);
	float pdf1, pdf2;
	glm::vec3 l;
	if (rnd < s) { // ggx importance sampler
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float cosTheta = glm::clamp(sqrtf((1.f - rnd) / (rnd * (a2 - 1.f) + 1)), 0.f, 1.f);
		float cosTheta2 = cosTheta * cosTheta;
		float sinTheta = sqrtf(1.f - cosTheta2);

		glm::vec3 h{ T * glm::cos(phi) * sinTheta + B * glm::sin(phi) * sinTheta + normal * cosTheta };
		h = glm::normalize(h);
		l = glm::reflect(v, h);

		// pdf of ggx
		float denom = ((a2 - 1) * cosTheta2 + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-v, h), 0.f) + FLT_EPSILON);

		// pdf of cosine hemisphere
		float cosine2 = fmaxf(glm::dot(l, normal), 0.f);
		pdf2 = sqrtf(1 - cosine2 * cosine2) * INV_PI;
	}
	else { // cosine hemisphere
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float r = sqrtf(rnd);

		glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };
		l = glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
		pdf2 = wo_tan.z * INV_PI;

		glm::vec3 h = halfway(-v, l);
		float cosTheta = glm::dot(l, normal);
		float denom = ((a2 - 1) * cosTheta * cosTheta + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-v, h), 0.f) + FLT_EPSILON);
	}

	pdf = s * pdf1 + (1.f - s) * pdf2;
	return l;
}

__device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getMetallic(intersect, mtl)
	getRoughness(intersect, mtl)
	getNormal(intersect, mtl)
	float s = 0.5f + metallic / 2.f;
	//float s = metallic;
	float a2 = roughness * roughness;

	glm::vec3 h = halfway(-v, l);
	float cosTheta = glm::dot(l, normal);
	float denom = ((a2 - 1) * cosTheta * cosTheta + 1);
	float pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-v, h), 0.f) + FLT_EPSILON);

	float cosine2 = fmaxf(glm::dot(l, normal), 0.f);
	float pdf2 = sqrtf(1 - cosine2 * cosine2) * INV_PI;

	return s * pdf1 + (1.f - s) * pdf2;
}

__device__ bool eval(
	const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng) {
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	getNormal(intersect, mtl)

	if (glm::dot(v, normal) >= 0.f) return false;

	getMetallic(intersect, mtl)
	getRoughness(intersect, mtl)
	getAlbedo(intersect, mtl)

	float s = 0.5f + metallic / 2.f;
	//float s = metallic;
	float a2 = roughness * roughness;

	glm::vec3 T = getDifferentDir(normal);
	T = glm::normalize(glm::cross(T, normal));
	glm::vec3 B = glm::normalize(glm::cross(T, normal));

	float rnd = u01(rng);
	float pdf, pdf1, pdf2;
	glm::vec3 h;
	float nh, nh2, denom;
	if (rnd < s) { // ggx importance sampler
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		nh = glm::clamp(sqrtf((1.f - rnd) / (rnd * (a2 - 1.f) + 1)), 0.f, 1.f);
		nh2 = nh * nh;
		float sinTheta = sqrtf(1.f - nh2);

		h = glm::vec3{ T * glm::cos(phi) * sinTheta + B * glm::sin(phi) * sinTheta + normal * nh };
		h = glm::normalize(h);
		l = glm::reflect(v, h);

		// pdf of ggx
		denom = ((a2 - 1) * nh2 + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-v, h), 0.f) + FLT_EPSILON);

		// pdf of cosine hemisphere
		float cosine2 = fmaxf(glm::dot(l, normal), 0.f);
		pdf2 = sqrtf(1 - cosine2 * cosine2) * INV_PI;
	}
	else { // cosine hemisphere
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float r = sqrtf(rnd);

		glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };
		l = glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
		pdf2 = wo_tan.z * INV_PI;

		h = halfway(-v, l);
		nh = glm::dot(h, normal);
		nh2 = nh * nh;
		denom = ((a2 - 1) * nh2 + 1);
		pdf1 = (a2)*INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-v, h), 0.f) + FLT_EPSILON);
	}

	if (glm::dot(l, normal) < 0.f) return false;
	pdf = s * pdf1 + (1.f - s) * pdf2;
	if (pdf < PDF_EPSILON) return false;

	float G, D;
	D = roughness / denom;
	D = D * D * INV_PI;

	float k = roughness * 0.5f;
	float G1 = fmaxf(glm::dot(normal, -v), FLT_EPSILON);
	G1 = G1 / (G1 * (1 - k) + k);
	float G2 = fmaxf(glm::dot(normal, l), FLT_EPSILON);
	G2 = G2 / (G2 * (1 - k) + k);
	G = G1 * G2;

	float hv = fmaxf(glm::dot(h, -v), 0.f);
	//glm::vec3 F = glm::mix(ior, albedo, metallic); // x * (1 - level) + y * level
	glm::vec3 F = albedo * metallic;
	F = F + (1.f - F) * powf(1.f - hv, 5.f);
	eval = (1.f - metallic) * (1.f - F) * albedo * INV_PI * fmaxf(glm::dot(l, normal), 0.f) +
		D * F * G * 0.25f / (fmaxf(glm::dot(-v, normal), FLT_EPSILON));
	eval /= pdf;
	return true;
}

}

namespace Specular {
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getNormal(intersect, mtl)
	if (glm::dot(v, normal) >= 0.f) return glm::vec3{ 0.f };
	getAlbedo(intersect, mtl)
	getMetallic(intersect, mtl)	
	float hv = fmaxf(glm::dot(normal, -v), 0.f);
	//glm::vec3 F = glm::mix(ior, albedo, metallic);
	glm::vec3 F = albedo * metallic;
	F = F + (1.f - F) * powf(1.f - hv, 5.f);

	return (1.f - metallic) * (1.f - F) * albedo * INV_PI * fmaxf(glm::dot(v, normal), 0.f) + F * albedo;
}

__device__ glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf) {
	getNormal(intersect, mtl)
	getMetallic(intersect, mtl)
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float hv = glm::dot(normal, -v);
	float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	float pdf2;
	glm::vec3 l;
	if (rnd > F) { // cosine hemisphere
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float r = sqrtf(rnd);

		glm::vec3 l_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };

		pdf2 = l_tan.z * INV_PI;

		glm::vec3 T = getDifferentDir(normal);
		T = glm::normalize(glm::cross(T, normal));
		glm::vec3 B = glm::normalize(glm::cross(T, normal));

		l = glm::normalize(T * l_tan.x + B * l_tan.y + normal * l_tan.z);
	}
	else {
		l = glm::reflect(v, normal);
		float cosine = fmaxf(glm::dot(l, normal), 0.f);
		pdf2 = sqrtf(1 - cosine * cosine) * INV_PI;
	}

	pdf = pdf2 * (1 - F) + F;
	return l;
}

__device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getNormal(intersect, mtl)
	getMetallic(intersect, mtl)
	float hv = glm::dot(normal, -v);
	float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	float cosine = fmaxf(glm::dot(l, normal), 0.f);
	return sqrtf(1 - cosine * cosine) * INV_PI * (1 - F) + F;
}

__device__ bool eval(
	const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng) {
	getNormal(intersect, mtl)
	if (glm::dot(v, normal) >= 0.f) return false;
	
	getMetallic(intersect, mtl)
	getAlbedo(intersect, mtl)
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);
	float rnd = u01(rng);
	float hv = glm::dot(normal, -v);
	float F = metallic + (1.f - metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);
	float pdf2;
	if (rnd > F) { // cosine hemisphere
		rnd = u01(rng);
		float phi = rnd * TWO_PI;
		rnd = u01(rng);
		float r = sqrtf(rnd);

		glm::vec3 l_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd) };

		pdf2 = l_tan.z * INV_PI;

		glm::vec3 T = getDifferentDir(normal);
		T = glm::normalize(glm::cross(T, normal));
		glm::vec3 B = glm::normalize(glm::cross(T, normal));

		l = glm::normalize(T * l_tan.x + B * l_tan.y + normal * l_tan.z);
	}
	else {
		l = glm::reflect(v, normal);
		float cosine = fmaxf(glm::dot(l, normal), 0.f);
		pdf2 = sqrtf(1 - cosine * cosine) * INV_PI;
	}

	if (glm::dot(l, normal) <= 0.f) return false;
	float pdf = pdf2 * (1 - F) + F;
	if (pdf < PDF_EPSILON) return false;
	//glm::vec3 F = glm::mix(ior, albedo, metallic);
	glm::vec3 F2 = albedo * metallic;
	F2 = F2 + (1.f - F2) * powf(1.f - hv, 5.f);

	eval = (1.f - metallic) * (1.f - F2) * albedo * INV_PI * fmaxf(glm::dot(v, normal), 0.f) + F2 * albedo;
	eval /= pdf;
	return true;
}

}

namespace Glass {
__device__ glm::vec3 bsdf(
	const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	getAlbedo(intersect, mtl)
	return albedo;
}

__device__ glm::vec3 sampler(
	const glm::vec3& v, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng, float& pdf) {
	getNormal(intersect, mtl)
	float ior = mtl.ior;
	
	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float invEta, iorI, iorT;
	float cosI = glm::dot(v, normal);
	pdf = 1.f;
	if (cosI < 0) { // enter
		cosI = -cosI;
		normal = -normal;
		invEta = 1.f / ior;
		iorI = 1.f;
		iorT = ior;
	}
	else { // leave
		invEta = ior;
		iorI = ior;
		iorT = 1.f;
	}

	float cosT = 1.f - invEta * invEta * fmaxf(0.f, (1.f - cosI * cosI));
	if (cosT <= 0.f) { // internal reflection
		return glm::reflect(v, -normal);
	}
	else {
		cosT = sqrtf(cosT);
		float F = fresnel(cosI, cosT, iorI, iorT);

		if (rnd < F) {
			return glm::reflect(v, -normal);
		}
		return v * invEta + normal * (cosT - invEta * cosI);
	}
}


__device__ float pdf(const glm::vec3& v, const glm::vec3& l, const IntersectInfo& intersect, const Material& mtl) {
	return 1.f;
}

__device__ bool eval(
	const glm::vec3& v, glm::vec3& l, glm::vec3& eval, const IntersectInfo& intersect, const Material& mtl, thrust::default_random_engine& rng) {
	getNormal(intersect, mtl)
	getAlbedo(intersect, mtl)
	float ior = mtl.ior;

	thrust::uniform_real_distribution<float> u01(0.f, 1.f);
	float rnd = u01(rng);
	float invEta, iorI, iorT;
	float cosI = glm::dot(v, normal);
	if (cosI < 0) { // enter
		cosI = -cosI;
		normal = -normal;
		invEta = 1.f / ior;
		iorI = 1.f;
		iorT = ior;
	}
	else { // leave
		invEta = ior;
		iorI = ior;
		iorT = 1.f;
	}

	float cosT = 1.f - invEta * invEta * fmaxf(0.f, (1.f - cosI * cosI));
	if (cosT <= 0.f) { // internal reflection
		l = glm::reflect(v, -normal);
	}
	else {
		cosT = sqrtf(cosT);
		float F = fresnel(cosI, cosT, iorI, iorT);

		if (rnd < F) {
			l = glm::reflect(v, -normal);
		}
		else l = v * invEta + normal * (cosT - invEta * cosI);
	}
	eval = albedo;
	return true;
}

}

}