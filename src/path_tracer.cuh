#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH

#include "common.cuh"
#include "intersection.cuh"
#include "sampler.cuh"
#include "bsdf.cuh"

namespace nagi {

// if the ray hit objects whose bounding boxes that are intersected with each other, we record up to 4 objects
constexpr int objIntersectTestBufSize = 4;
// if the ray hit triangles whose bounding boxes that are intersected with each other, we record up to 16 triangles
constexpr int trigIntersectTestBufSize = 16;

struct IntersectInfo {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
	int mtlIdx;
};
struct ifHit {
	__host__ __device__ bool operator()(const Path& x) {
		return x.lastHit >= 0;
	}
};
struct ifNotHit {
	__host__ __device__ bool operator()(const Path& x) {
		return x.lastHit < 0;
	}
};
struct ifTerminated {
	__host__ __device__ bool operator()(const Path& x) {
		return x.remainingBounces <= 0;
	}
};
struct ifNotTerminated {
	__host__ __device__ bool operator()(const Path& x) {
		return x.remainingBounces > 0;
	}
};
struct ifNonNegtive {
	__host__ __device__ bool operator()(const int& x) {
		return x >= 0;
	}
}; 
struct ifNegtive {
	__host__ __device__ bool operator()(const int& x) {
		return x < 0;
	}
}; 
struct IntersectionComp {
	__host__ __device__ bool operator()(const IntersectInfo& a, const IntersectInfo& b) {
		return (a.mtlIdx < b.mtlIdx);
	}
};
//struct ObjectIntersectInfo {
//	int objIdx[objIntersectTestBufSize];
//	int num;
//};
//struct TriangleIntersectInfo {
//	int trigIdx[trigIntersectTestBufSize];
//	glm::vec3 normals[trigIntersectTestBufSize];
//	int num;
//};

__global__ void kernInitializeFrameBuffer(float* frame);
__global__ void kernInitializeRays(int spp, Path* rayPool, int maxBounce, const Camera cam);
//__global__ void kernObjIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, int* hitMtlIdx);
__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out);
__global__ void kernShading(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernWriteFrameBuffer(float spp, Path* rayPool, float* frameBuffer);

class PathTracer {
public:
	PathTracer(Scene& Scene) :scene{ Scene } {}
	~PathTracer();
	PathTracer(const PathTracer&) = delete;
	void operator=(const PathTracer&) = delete;

	void initialize();
	void allocateBuffers();
	void destroyBuffers();
	void iterate();

	int intersectionTest(int rayNum);
	// sort rays according to materials
	void sortRays(int rayNum);
	// compute color and generate new rays
	int shade(int rayNum, int spp);

	// delete rays whose flag is negetive
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool);
	// delete rays that didn't hit triangles
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults);

	void writeFrameBuffer();

	std::unique_ptr<float[]> getFrameBuffer();
	const float* const getDevFrameBuffer() const { return devFrameBuf; }

	bool printDetails{ false };
	Scene& scene;

	// ping-pong buffers
	Path* devRayPool1{ nullptr };
	Path* devRayPool2{ nullptr };

	IntersectInfo* devResults1{ nullptr };
	IntersectInfo* devResults2{ nullptr };

	Path* devTerminatedRays{ nullptr };
	int terminatedRayNum{ 0 };

	Object* devObjBuf{ nullptr };
	Material* devMtlBuf{ nullptr };
	Triangle* devTrigBuf{ nullptr };
	// The frame buffer is a rectangle of pixels stored from left-to-right, top-to-bottom.
	float* devFrameBuf{ nullptr };
};

}

#endif // !PATH_TRACER_CUH
