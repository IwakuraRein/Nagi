#include "intersection.cuh"

#include <glm/gtx/intersect.hpp>

namespace nagi {

// with reference to https://tavianator.com/2011/ray_box.html
__device__ __host__ bool rayBoxIntersect(const Ray& r, const BoundingBox& bbox, float* dist) {
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

__device__ __host__ bool rayTrigIntersect(
	const Ray& r, const Triangle& triangle, float* dist, glm::vec3* normal, glm::vec2* uv) {
	glm::vec2 baryCoord;
	if (glm::intersectRayTriangle(
		r.origin, r.dir, triangle.vert0.position, triangle.vert1.position, triangle.vert2.position, baryCoord, *dist)) {
		*normal = baryCoord.x * triangle.vert0.normal + baryCoord.y * triangle.vert1.normal + (1 - baryCoord.x - baryCoord.y) * triangle.vert2.normal;
		if (glm::dot(*normal, r.dir) < 0) { // angle > 90
			// uv needs another interpolation order. don't know why
			//*uv = baryCoord.x * triangle.vert0.uv + baryCoord.y * triangle.vert1.uv + (1 - baryCoord.x - baryCoord.y) * triangle.vert2.uv;
			*uv = (1 - baryCoord.x - baryCoord.y) * triangle.vert0.uv + baryCoord.x * triangle.vert1.uv + baryCoord.y * triangle.vert2.uv;
			return true;
		}
		else return false;
	}
	else return false;
}

__device__ __host__ bool boxBoxIntersect(const BoundingBox& a, const BoundingBox& b/*, BoundingBox* overlap*/) {
	//return ((vec3Comp(b1.min, b2.min) && vec3Comp(b2.min, b1.max)) ||
	//	(vec3Comp(b1.min, b2.max) && vec3Comp(b2.max, b1.max)) ||
	//	(vec3Comp(b2.min, b1.min) && vec3Comp(b1.min, b2.max)) ||
	//	(vec3Comp(b2.min, b1.max) && vec3Comp(b1.max, b2.max)));
	const float dx1 = a.min.x - b.max.x;
	const float dx2 = b.min.x - a.max.x;
	const float dy1 = a.min.y - b.max.y;
	const float dy2 = b.min.y - a.max.y;
	const float dz1 = a.min.z - b.max.z;
	const float dz2 = b.min.z - a.max.z;

	return (dx1 <= 0 && dx2 <= 0) &&
		(dy1 <= 0 && dy2 <= 0) &&
		(dz1 <= 0 && dz2 <= 0);
}

// reference: https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/aabb-triangle.html
__device__ __host__ bool tirgBoxIntersect(const Triangle& t, const BoundingBox& b) {
	glm::vec3 v0 = t.vert0.position - b.center;
	glm::vec3 v1 = t.vert1.position - b.center;
	glm::vec3 v2 = t.vert2.position - b.center;
	glm::vec3 e = b.halfExtent * 2.f;

	const glm::vec3 f0{ v0 - v1};
	const glm::vec3 f1{ v1 - v2};
	const glm::vec3 f2{ v2 - v0};

	const glm::vec3 u0{ 1.0f, 0.0f, 0.0f };
	const glm::vec3 u1{ 0.0f, 1.0f, 0.0f };
	const glm::vec3 u2{ 0.0f, 0.0f, 1.0f };

	// We first test against 9 axis, these axis are given by
	// cross product combinations of the edges of the triangle
	// and the edges of the AABB.

	const glm::vec3 axis_u0_f0 = glm::cross(u0, f0);
	const glm::vec3 axis_u0_f1 = glm::cross(u0, f1);
	const glm::vec3 axis_u0_f2 = glm::cross(u0, f2);

	const glm::vec3 axis_u1_f0 = glm::cross(u1, f0);
	const glm::vec3 axis_u1_f1 = glm::cross(u1, f1);
	const glm::vec3 axis_u1_f2 = glm::cross(u2, f2);

	const glm::vec3 axis_u2_f0 = glm::cross(u2, f0);
	const glm::vec3 axis_u2_f1 = glm::cross(u2, f1);
	const glm::vec3 axis_u2_f2 = glm::cross(u2, f2);

	// Testing axis: axis_u0_f0
	float p0 = glm::dot(t.vert0.position, axis_u0_f0);
	float p1 = glm::dot(t.vert1.position, axis_u0_f0);
	float p2 = glm::dot(t.vert2.position, axis_u0_f0);
	float r = e.x * fabsf(glm::dot(u0, axis_u0_f0)) +
		e.y * fabsf(glm::dot(u1, axis_u0_f0)) +
		e.z * fabsf(glm::dot(u2, axis_u0_f0));
	if (fmaxf(-fmaxf(p0, p1, p2), fminf(p0, p1, p2)) > r) {
		// This means BOTH of the points of the projected triangle
		// are outside the projected half-length of the AABB
		// Therefore the axis is seperating and we can exit
		return false;
	}
	return true;
}

}