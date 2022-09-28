#ifndef INTERSECTION_CUH
#define INTERSECTION_CUH

#include "common.cuh"

namespace nagi {

inline __device__ bool vec3Comp(const glm::vec3& vec1, const glm::vec3& vec2) {
	return (vec1.x < vec2.x) && (vec1.y < vec2.y) && (vec1.z < vec2.z);
}

__device__ bool rayBoxIntersect(const Ray& r, const BoundingBox& bbox, float* dist);

__device__ bool rayTrigIntersect(const Ray& r, const Triangle& triangle, float* dist, glm::vec3* normal);

__device__ bool boxBoxIntersect(const BoundingBox& b1, const BoundingBox& b2);

}


#endif // !INTERSECTION_CUH
