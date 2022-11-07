#ifndef DENOISER_CUH
#define DENOISER_CUH

#include "common.cuh"
#include "path_tracer.cuh"

namespace nagi {

class Denoiser {
public:
	Denoiser(Scene& Scene, PathTracer& PathTracer) :
		scene{ Scene }, window{ scene.window }, pathTracer{ PathTracer } {}
	~Denoiser();
	Denoiser(const Denoiser&) = delete;
	void operator=(const Denoiser&) = delete;

	std::unique_ptr<float[]> denoise();

	Scene& scene;
	WindowSize& window;
	PathTracer& pathTracer;

	float* devDenoised{ nullptr };
};


}

#endif // !DENOISER_CUH
