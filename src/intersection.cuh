#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "common.cuh"

namespace nagi {

__device__ constexpr float boxNormals[3][3]{
	{ 1.f, 0.f, 0.f },
	{ 0.f, 1.f, 0.f },
	{ 0.f, 0.f, 1.f }
};

inline __device__ __host__ glm::vec2 trigProject(const Triangle& trig, const glm::vec3& dir) {
	glm::vec2 out{ FLT_MAX, -FLT_MAX };
	float val = glm::dot(trig.vert0.position, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(trig.vert1.position, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(trig.vert2.position, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	return out;
}

inline __device__ __host__ glm::vec2 boxProject(const BoundingBox& b, const glm::vec3& dir) {
	glm::vec2 out{ FLT_MAX, -FLT_MAX };
	glm::vec3 extent = b.max - b.min;
	float val = glm::dot(b.min, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.max, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.x, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.y, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.z, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.x+extent.y, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.x+extent.z, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	val = glm::dot(b.min+extent.y+extent.z, dir);
	if (val < out.x) out.x = val;
	if (val > out.y) out.y = val;
	return out;
}

inline __device__ __host__ bool vec3Comp(const glm::vec3& vec1, const glm::vec3& vec2) {
	return (vec1.x <= vec2.x) && (vec1.y <= vec2.y) && (vec1.z <= vec2.z);
}

__device__ __host__ bool rayBoxIntersect(const Ray& r, const BoundingBox& bbox, float* dist);

__device__ __host__ bool rayTrigIntersect(const Ray& r, const Triangle& triangle, float* dist, glm::vec3* normal, glm::vec2* uv);

__device__ __host__ bool boxBoxIntersect(const BoundingBox& b1, const BoundingBox& b2/*, BoundingBox* overlap*/);

__device__ __host__ bool tirgBoxIntersect(const Triangle& triangle, const BoundingBox& bbox);

}


#endif // !INTERSECTION_CUH
