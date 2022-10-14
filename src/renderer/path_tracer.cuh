#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH

#include "common.cuh"
#include "intersection.cuh"
#include "bvh.cuh"
#include "sampler.cuh"
#include "bsdf.cuh"

#define PDF_EPSILON 0.0001f
#define MAX_GBUFFER_BOUNCE 8
#define REFLECT_OFFSET 0.0002f
#define REFRACT_OFFSET 0.001f

namespace nagi {

struct IntersectInfo {
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	glm::vec3 position;
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
__global__ void kernInitializeRays(
WindowSize window, int spp, Path* rayPool, int maxBounce, const Camera cam/*, bool jitter = true, bool DOP = true*/);
// naive bvh
__global__ void kernObjIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, Triangle* trigBuf, IntersectInfo* out);
// brute force
__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out);
// oct tree bvh
__global__ void kernBVHIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, BVH::Node* treeBuf, int* treeTrigBuf, Triangle* trigBuf, IntersectInfo* out);
__global__ void kernShadeLightSource(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernShadeLambert(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernShadeSpecular(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernShadeGlass(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernShadeMicrofacet(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf);
__global__ void kernWriteFrameBuffer(WindowSize window, float currentSpp, Path* rayPool, float* frameBuffer);
__global__ void kernGenerateGbuffer(
	int rayNum, float currentSpp, int bounce, glm::vec3 camPos, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf,
	float* currentAlbedoBuf, /*float* currentNormalBuf, */float* currentDepthBuf, float* albedoBuf, float* normalBuf, float* depthBuf);
__global__ void kernShadeWithSkybox(int rayNum, cudaTextureObject_t skybox, glm::vec3 rotate, glm::vec3 up, glm::vec3 right, Path* rayPool);
__global__ void kernGenerateSkyboxAlbedo(
	int rayNum, float currentSpp, cudaTextureObject_t skybox, glm::vec3 rotate, glm::vec3 up, glm::vec3 right, Path* rayPool, float* albedoBuf);

class PathTracer {
public:
	PathTracer(Scene& Scene, BVH& BVH) :scene{ Scene }, window{ scene.window }, bvh{ BVH } {}
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

	void generateGbuffer(int rayNum, int spp, int bounce);
	void generateSkyboxAlbedo(int rayNum, int spp);

	// delete rays whose flag is negetive
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool);
	// delete rays that didn't hit triangles
	int compactRays(int rayNum, Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults);

	void writeFrameBuffer(int spp);

	void shadeWithSkybox();

	bool finish() const { return spp > scene.config.spp; }

	std::unique_ptr<float[]> getFrameBuffer();
	void copyFrameBuffer(float* frameBuffer);
	std::unique_ptr<float[]> getNormalBuffer();
	std::unique_ptr<float[]> getAlbedoBuffer();
	std::unique_ptr<float[]> getDepthBuffer();

	bool printDetails{ false };
	Scene& scene;
	BVH& bvh;
	WindowSize& window;

	// ping-pong buffers
	Path* devRayPool1{ nullptr };
	Path* devRayPool2{ nullptr };

	IntersectInfo* devResults1{ nullptr };
	IntersectInfo* devResults2{ nullptr };

	Path* devTerminatedRays{ nullptr };
	int terminatedRayNum{ 0 }, spp{ 1 };

	Object* devObjBuf{ nullptr };
	Material* devMtlBuf{ nullptr };
	Triangle* devTrigBuf{ nullptr };
	// The frame buffer is a rectangle of pixels stored from left-to-right, top-to-bottom.
	float* devFrameBuf{ nullptr };
	float* devNormalBuf{ nullptr };
	float* devAlbedoBuf{ nullptr };
	float* devDepthBuf{ nullptr };
	//float* devCurrentNormalBuf{ nullptr };
	float* devCurrentAlbedoBuf{ nullptr };
	float* devCurrentDepthBuf{ nullptr };
};

}

#endif // !PATH_TRACER_CUH
