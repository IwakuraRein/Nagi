#ifndef PATH_TRACER_HPP
#define PATH_TRACER_HPP

#include "common.cuh"
#include "intersection.cuh"

namespace nagi {

// if the ray hit objects whose bounding boxes that are intersected with each other, we record up to 4 objects
constexpr int objIntersectTestBufSize = 4;
// if the ray hit triangles whose bounding boxes that are intersected with each other, we record up to 16 triangles
constexpr int trigIntersectTestBufSize = 16;

struct ObjectIntersectInfo {
	int objIdx[objIntersectTestBufSize];
	int num;
};
struct TriangleIntersectInfo {
	int trigIdx[trigIntersectTestBufSize];
	glm::vec3 normals[trigIntersectTestBufSize];
	int num;
};

__global__ void kernInitializeFrameBuffer(float* frame);
__global__ void kernInitializeRays(Path* rayPool, int maxBounce, const Camera cam);
//__global__ void kernObjIntersectTest(Path* rayPool, Object* objBuf, ObjectIntersectInfo* out, int* flags);
//__global__ void kernTrigIntersectTest(Path* rayPool, ObjectIntersectInfo* objBuf, Triangle* trigBuf, TriangleIntersectInfo* out, int* flags);

class PathTracer {
public:
	PathTracer(Scene& Scene) :scene{ Scene } {}
	~PathTracer();
	PathTracer(const PathTracer&) = delete;
	void operator=(const PathTracer&) = delete;

	void initialize();
	bool finished() const { return true; }
	void iterate() {}
	void compactRays();

	void test1(float* frameBuffer);

	bool printDetails{ false };
	Scene& scene;
	Path* devRayPool{ nullptr };

	Object* devObjBuf{ nullptr };
	Material* devMtlBuf{ nullptr };
	Triangle* devTrigBuf{ nullptr };
	// The frame buffer is a rectangle of pixels stored from left-to-right, top-to-bottom.
	float* devFrameBuf;
};

}

#endif // !PATH_TRACER_HPP
