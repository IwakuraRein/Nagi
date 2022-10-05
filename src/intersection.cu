#include "intersection.cuh"

#include <glm/gtx/intersect.hpp>

namespace nagi {

// with reference to https://tavianator.com/2011/ray_box.html
__device__ __host__ bool rayBoxIntersect(const Ray& r, const glm::vec3& min, const glm::vec3& max, float* dist) {
	float tx1 = (min.x - r.origin.x) * r.invDir.x;
	float tx2 = (max.x - r.origin.x) * r.invDir.x;

	float tmin = fminf(tx1, tx2);
	float tmax = fmaxf(tx1, tx2);

	float ty1 = (min.y - r.origin.y) * r.invDir.y;
	float ty2 = (max.y - r.origin.y) * r.invDir.y;

	tmin = fmaxf(tmin, fminf(ty1, ty2));
	tmax = fminf(tmax, fmaxf(ty1, ty2));

	float tz1 = (min.z - r.origin.z) * r.invDir.z;
	float tz2 = (max.z - r.origin.z) * r.invDir.z;

	tmin = fmaxf(tmin, fminf(tz1, tz2));
	tmax = fminf(tmax, fmaxf(tz1, tz2));

	if (tmax > 0 && tmax > tmin) {
		*dist = tmin;
		return true;
	}
	else return false;
}
__device__ __host__ bool rayBoxIntersect(const Ray& r, const BoundingBox& bbox, float* dist) {
	return rayBoxIntersect(r, bbox.min, bbox.max, dist);
}

__device__ __host__ bool rayTrigIntersect(
	const Ray& r, const Triangle& triangle, float* dist, glm::vec3* normal, glm::vec3* tangent, glm::vec2* uv) {
	glm::vec2 baryCoord;
	if (glm::intersectRayTriangle(
		r.origin, r.dir, triangle.vert0.position, triangle.vert1.position, triangle.vert2.position, baryCoord, *dist)) {
		*normal = (1 - baryCoord.x - baryCoord.y) * triangle.vert0.normal + baryCoord.x * triangle.vert1.normal + baryCoord.y * triangle.vert2.normal;
		*uv = (1 - baryCoord.x - baryCoord.y) * triangle.vert0.uv + baryCoord.x * triangle.vert1.uv + baryCoord.y * triangle.vert2.uv;
		*tangent = (1 - baryCoord.x - baryCoord.y) * triangle.vert0.tangent + baryCoord.x * triangle.vert1.tangent + baryCoord.y * triangle.vert2.tangent;
		return true;
	}
	return false;
}

__device__ __host__ bool boxBoxIntersect(const BoundingBox& a, const BoundingBox& b/*, BoundingBox* overlap*/) {
	//return ((vec3Comp(b1.min, b2.min) && vec3Comp(b2.min, b1.max)) ||
	//	(vec3Comp(b1.min, b2.max) && vec3Comp(b2.max, b1.max)) ||
	//	(vec3Comp(b2.min, b1.min) && vec3Comp(b1.min, b2.max)) ||
	//	(vec3Comp(b2.min, b1.max) && vec3Comp(b1.max, b2.max)));

	const glm::vec3 amin{ a.min - b.center };
	const glm::vec3 amax{ a.max - b.center };
	if (amin.x > b.halfExtent.x || amax.x < -b.halfExtent.x ||
		amin.y > b.halfExtent.y || amax.y < -b.halfExtent.y ||
		amin.z > b.halfExtent.z || amax.z < -b.halfExtent.z)
		return false;

	return true;
}

// reference: https://gdbooks.gitbooks.io/3dcollisions/content/Chapter4/aabb-triangle.html
//            https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox2.txt
inline __device__ __host__ bool axisProjectTest(
	const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, 
	const glm::vec3& u0, const glm::vec3& u1, const glm::vec3& u2, 
	const glm::vec3& e, const glm::vec3& axis) {
	float p0 = glm::dot(v0, axis);
	float p1 = glm::dot(v1, axis);
	float p2 = glm::dot(v2, axis);
	float r = e.x * fabsf(glm::dot(u0, axis)) +
		e.y * fabsf(glm::dot(u1, axis)) +
		e.z * fabsf(glm::dot(u2, axis));

	// This means BOTH of the points of the projected triangle
	// are outside the projected half-length of the AABB
	// Therefore the axis is seperating and we can exit
	return fmaxf(-fmaxf(p0, p1, p2), fminf(p0, p1, p2)) > r;
}
__device__ __host__ bool tirgBoxIntersect(const Triangle& t, const BoundingBox& b) {
	glm::vec3 v0 = t.vert0.position - b.center;
	glm::vec3 v1 = t.vert1.position - b.center;
	glm::vec3 v2 = t.vert2.position - b.center;
	glm::vec3 extent = b.max - b.min;

	const glm::vec3 e0{ v1 - v0 };
	const glm::vec3 e1{ v2 - v1 };
	const glm::vec3 e2{ v0 - v2 };
	const glm::vec3 n{ glm::cross(e0, e1) };

	const glm::vec3 u0{ 1.0f, 0.0f, 0.0f };
	const glm::vec3 u1{ 0.0f, 1.0f, 0.0f };
	const glm::vec3 u2{ 0.0f, 0.0f, 1.0f };

	// We first test against 9 axis, these axis are given by
	// cross product combinations of the edges of the triangle
	// and the edges of the AABB.

	const glm::vec3 axis_u0_e0 = glm::cross(u0, e0);
	const glm::vec3 axis_u0_e1 = glm::cross(u0, e1);
	const glm::vec3 axis_u0_e2 = glm::cross(u0, e2);

	const glm::vec3 axis_u1_e0 = glm::cross(u1, e0);
	const glm::vec3 axis_u1_e1 = glm::cross(u1, e1);
	const glm::vec3 axis_u1_e2 = glm::cross(u2, e2);

	const glm::vec3 axis_u2_e0 = glm::cross(u2, e0);
	const glm::vec3 axis_u2_e1 = glm::cross(u2, e1);
	const glm::vec3 axis_u2_e2 = glm::cross(u2, e2);

	// test 9 axis
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u0_e0)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u0_e1)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u0_e2)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u1_e0)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u1_e1)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u1_e2)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u2_e0)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u2_e1)) return false;
	if (axisProjectTest(v0, v1, v2, u0, u1, u2, extent, axis_u2_e2)) return false;

	// test 3 box normals. equivalent to test two aabb
	if (boxBoxIntersect(t.bbox, b)) {

		// test if the box intersects the plane of the triangle
		// compute plane equation of triangle: normal*x+d=0
		float d = -glm::dot(n, v0);
		glm::vec3 vmin, vmax;
#pragma unroll
		for (int q = 0; q <= 2; q++) {
			if (n[q] > 0.f) {
				vmin[q] = -b.halfExtent[q];
				vmax[q] = b.halfExtent[q];
			}
			else {
				vmin[q] = b.halfExtent[q];
				vmax[q] = -b.halfExtent[q];
			}
		}
		if (glm::dot(n, vmax) + d >= 0.f) return true;
	}

	return false;
}

}