#ifndef PATH_TRACER_CUH
#define PATH_TRACER_CUH

#include "common.cuh"
#include "intersection.cuh"
#include "bvh.cuh"
#include "sampler.cuh"
#include "bsdf.cuh"
#include "material.cuh"

#define MAX_GBUFFER_BOUNCE 8
#define REFLECT_OFFSET 0.0002f
#define REFRACT_OFFSET 0.001f

namespace nagi {

class PathTracer {
public:
	PathTracer(Scene& Scene, BVH& BVH);
	~PathTracer();
	PathTracer(const PathTracer&) = delete;
	void operator=(const PathTracer&) = delete;

	void allocateBuffers();
	void destroyBuffers();
	void iterate();

	int intersectionTest(int rayNum);
	// sort rays according to materials
	void sortRays(int rayNum);
	// compute color and generate new rays
	int shade(int rayNum, int spp, int bounce);

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
	float* devCurrentNormalBuf{ nullptr };
	float* devCurrentAlbedoBuf{ nullptr };
	float* devCurrentDepthBuf{ nullptr };
	float* devLumiance2Buf{ nullptr };


	cudaEvent_t timer_start{ nullptr }, timer_end{ nullptr };
	bool timerStarted{ false };
	void tik() {
		cudaEventRecord(timer_start);
		timerStarted = true;
	}
	float tok() {
		if (!timerStarted) return 0.f;
		cudaEventRecord(timer_end);
		cudaEventSynchronize(timer_end);
		float t;
		cudaEventElapsedTime(&t, timer_start, timer_end);
		timerStarted = false;
		return t;
	}
	float intersectionTime{ 0.f }, compactionTime{ 0.f }, shadingTime{ 0.f }, gbufferTime{ 0.f };
};

}

#endif // !PATH_TRACER_CUH
