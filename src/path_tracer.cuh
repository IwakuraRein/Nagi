#ifndef PATH_TRACER_HPP
#define PATH_TRACER_HPP

#include "common.cuh"
#include "intersection.cuh"

namespace nagi {

// if the ray hit objects whose bounding boxes that are intersected with each other, we record up to 4 objects
constexpr int objIntersectTestBufSize = 4;
// if the ray hit triangles whose bounding boxes that are intersected with each other, we record up to 16 triangles
constexpr int trigIntersectTestBufSize = 16;

struct IntersectInfo {
	bool hit;
	glm::vec3 position;
	glm::vec3 normal;
	int mtlId;
};
struct ifHit {
	__host__ __device__ bool operator()(const IntersectInfo& x) {
		return x.hit;
	}
};
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
	void allocateBuffers();
	void destroyBuffers();
	bool finished() const { return spp == scene.config.spp; }
	void iterate();
	void intersectionTest(Path* rayPool, IntersectInfo* results);
	// delete unhited rays
	int compactRays(Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults);
	// delete rays that hit lights
	int compactRays(Path* rayPool, Path* compactedRayPool, int* flags);
	// sort rays according to materials
	void sort(Path* rayPool, Path* sortedRayPool, IntersectInfo* intersectResults, IntersectInfo* sortedIntersectResults);
	// compute color and generate new rays
	void shade(IntersectInfo* intersectResults, int* flags);

	void test1(float* frameBuffer);

	int spp{ 0 };
	int bounce{ 0 };
	int remainingRays;

	bool printDetails{ false };
	Scene& scene;

	// ping-pong buffers
	Path* devRayPool1{ nullptr };
	Path* devRayPool2{ nullptr };

	IntersectInfo* devResults1{ nullptr };
	IntersectInfo* devResults2{ nullptr };

	Object* devObjBuf{ nullptr };
	Material* devMtlBuf{ nullptr };
	Triangle* devTrigBuf{ nullptr };
	int* devShadeFlags{ nullptr };
	// The frame buffer is a rectangle of pixels stored from left-to-right, top-to-bottom.
	float* devFrameBuf;
};

}

#endif // !PATH_TRACER_HPP
