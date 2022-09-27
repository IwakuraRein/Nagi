#include "intersection.cuh"

#include <glm/gtx/intersect.hpp>

namespace nagi {

	__device__ bool rayBoxIntersect(const Ray r, const BoundingBox bbox, float* dist) {
		// assume r.dir will never have 0
		glm::vec3 tmin0 = (bbox.min - r.origin) / r.dir;
		glm::vec3 tmax0 = (bbox.max - r.origin) / r.dir;
		float tmin = glm::max(tmin0.x, glm::max(tmin0.y, tmin0.z));
		float tmax = glm::min(tmax0.x, glm::min(tmax0.y, tmax0.z));

		if (tmax > 0 && tmax > tmin) {
			*dist = tmin;
			return true;
		}
		else return false;
	}

	__device__ bool rayTrigIntersect(
		const Ray r, const Triangle triangle, float* dist, glm::vec3* normal) {
		glm::vec2 baryCoord;
		if (glm::intersectRayTriangle(
				r.origin, r.dir, triangle.vert0.position, triangle.vert1.position, triangle.vert2.position, baryCoord, *dist)) {
			*normal = baryCoord.x * triangle.vert0.normal + baryCoord.y * triangle.vert1.normal + (1 - baryCoord.x - baryCoord.y) * triangle.vert2.normal;
			if (glm::dot(*normal, r.dir) < 0) // angle > 90
				return true;
			else return false;
		}
		else return false;
	}

	__device__ bool boxBoxIntersect(const BoundingBox b1, const BoundingBox b2) {
		return ((vec3Comp(b1.min, b2.min) && vec3Comp(b2.min, b1.max)) ||
			(vec3Comp(b1.min, b2.max) && vec3Comp(b2.max, b1.max)) ||
			(vec3Comp(b2.min, b1.min) && vec3Comp(b1.min, b2.max)) ||
			(vec3Comp(b2.min, b1.max) && vec3Comp(b1.max, b2.max)));
	}

}