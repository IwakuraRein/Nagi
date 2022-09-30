#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH

#include "common.cuh"
#include "intersection.cuh"
#include "sampler.cuh"
#include "bsdf.cuh"

namespace nagi {

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
struct IntersectionComp {
	__host__ __device__ bool operator()(const IntersectInfo& a, const IntersectInfo& b) {
		return (a.mtlIdx < b.mtlIdx);
	}
};

__global__ void kernInitializeFrameBuffer(WindowSize window, float* frame);
__global__ void kernInitializeRays(WindowSize window, int spp, Path* rayPool, int maxBounce, const Camera cam, bool jitter);
__global__ void kernIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, Triangle* trigBuf, IntersectInfo* out);
__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out);
__global__ void kernShading(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernWriteFrameBuffer(WindowSize window, float currentSpp, Path* rayPool, float* frameBuffer);
__global__ void kernGenerateGbuffer(int rayNum, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf, float* albedoBuf, float* normalBuf, float* depthBuf);

class PathTracer {
public:
	PathTracer(Scene& Scene) :scene{ Scene }, window{ Scene.config.window } {}
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

	void generateGbuffer(int rayNum);

	// delete rays whose flag is negetive
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool);
	// delete rays that didn't hit triangles
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults);

	void writeFrameBuffer(int spp);

	std::unique_ptr<float[]> getFrameBuffer();
	void copyFrameBuffer(float* frameBuffer);
	std::unique_ptr<float[]> getNormalBuffer();
	std::unique_ptr<float[]> getAlbedoBuffer();
	std::unique_ptr<float[]> getDepthBuffer();

	bool printDetails{ false };
	Scene& scene;
	WindowSize& window;
	bool hasGbuffer{ false };

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
	float* devNormalBuf{ nullptr };
	float* devAlbedoBuf{ nullptr };
	float* devDepthBuf{ nullptr };
};

}

#endif // !PATH_TRACER_CUH
