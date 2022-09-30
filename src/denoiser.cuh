#ifndef DENOISER_CUH
#define DENOISER_CUH

#include "common.cuh"
#include "path_tracer.cuh"

#define FILTER_SIZE 17
#define FILTER_SIZE_HALF 8
#define FILTER_SIZE_SQUARE 289

namespace nagi {

__global__ void kernGenerateLuminance(WindowSize window, float* color, float* albedo);
__global__ void kernRetrieveColor(WindowSize window, float* luminance, float* albedo);

// 17x17 bilateral filter
// reference: https://dl.acm.org/doi/10.1145/3105762.3105770; http://diglib.eg.org/handle/10.2312/EGGH.HPG10.067-075
__global__ void kernBilateralFilter(
	WindowSize window, float* denoised, float* luminance, float* normal, float* depth, float sigmaN, float sigmaD, float sigmaL);

class Denoiser {
public:
	Denoiser(Scene& Scene, PathTracer& PathTracer) :
		scene{ Scene }, window{ Scene.config.window }, pathTracer{ PathTracer } {}
	~Denoiser();
	Denoiser(const Denoiser&) = delete;
	void operator=(const Denoiser&) = delete;

	std::unique_ptr<float[]> denoise();

	std::unique_ptr<float[]> bilateralFilter(float* frameBuf, float* albedoBuf, float* normalBuf, float* depthBuf);
	std::unique_ptr<float[]> openImageDenoiser(float* frameBuf, float* albedoBuf, float* normalBuf);

	Scene& scene;
	WindowSize& window;
	PathTracer& pathTracer;

	float* devDenoised{ nullptr };
};


}

#endif // !DENOISER_CUH
