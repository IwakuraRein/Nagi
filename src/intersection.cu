#include "intersection.cuh"

#include <glm/gtx/intersect.hpp>

namespace nagi {

	// with reference to https://tavianator.com/2011/ray_box.html
	__device__ bool rayBoxIntersect(const Ray& r, const BoundingBox& bbox, float* dist) {
		float tx1 = (bbox.min.x - r.origin.x) * r.invDir.x;
		float tx2 = (bbox.max.x - r.origin.x) * r.invDir.x;
		
		float tmin = fminf(tx1, tx2);
		float tmax = fmaxf(tx1, tx2);
		
		float ty1 = (bbox.min.y - r.origin.y) * r.invDir.y;
		float ty2 = (bbox.max.y - r.origin.y) * r.invDir.y;

		tmin = fmaxf(tmin, fminf(ty1, ty2));
		tmax = fminf(tmax, fmaxf(ty1, ty2));

		float tz1 = (bbox.min.z - r.origin.z) * r.invDir.z;
		float tz2 = (bbox.max.z - r.origin.z) * r.invDir.z;

		tmin = fmaxf(tmin, fminf(tz1, tz2));
		tmax = fminf(tmax, fmaxf(tz1, tz2));

		if (tmax > 0 && tmax > tmin) {
			*dist = tmin;
			return true;
		}
		else return false;
	}

	__device__ bool rayTrigIntersect(
		const Ray& r, const Triangle& triangle, float* dist, glm::vec3* normal) {
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

	__device__ bool boxBoxIntersect(const BoundingBox& b1, const BoundingBox& b2) {
		return ((vec3Comp(b1.min, b2.min) && vec3Comp(b2.min, b1.max)) ||
			(vec3Comp(b1.min, b2.max) && vec3Comp(b2.max, b1.max)) ||
			(vec3Comp(b2.min, b1.min) && vec3Comp(b1.min, b2.max)) ||
			(vec3Comp(b2.min, b1.max) && vec3Comp(b1.max, b2.max)));
	}

}