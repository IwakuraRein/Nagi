#ifndef DENOISER_CUH
#define DENOISER_CUH

#include "common.cuh"
#include "path_tracer.cuh"

#define FILTER_SIZE 9
#define FILTER_SIZE_HALF 4
#define FILTER_AREA 81
#define FILTER_AREA_HALF 40

namespace nagi {

// CUDA 11.3 has added device code support for new C++ keywords: `constexpr` and `auto`.
// In CUDA C++, `__device__` and `__constant__` variables can now be declared `constexpr`.
// The constexpr variables can be used in constant expressions, where they are evaluated at
// compile time, or as normal variables in contexts where constant expressions are not required.
//__device__ constexpr float devGaussianKernel[25]{
//	1.f/* / 273.f*/, 4.f /* / 273.f*/, 7.f /* / 273.f*/, 4.f /* / 273.f*/, 1.f/* / 273.f*/,
//	4.f/* / 273.f*/, 16.f/* / 273.f*/, 26.f/* / 273.f*/, 16.f/* / 273.f*/, 4.f/* / 273.f*/,
//	7.f/* / 273.f*/, 26.f/* / 273.f*/, 41.f/* / 273.f*/, 26.f/* / 273.f*/, 7.f/* / 273.f*/,
//	4.f/* / 273.f*/, 16.f/* / 273.f*/, 26.f/* / 273.f*/, 16.f/* / 273.f*/, 4.f/* / 273.f*/,
//	1.f/* / 273.f*/, 4.f /* / 273.f*/, 7.f /* / 273.f*/, 4.f /* / 273.f*/, 1.f/* / 273.f*/
//};

__device__ constexpr float devGaussianKernel[81] {
	1.f / 65536.f, 8.f / 65536.f, 28.f / 65536.f, 56.f / 65536.f, 70.f / 65536.f, 56.f / 65536.f, 28.f / 65536.f, 8.f / 65536.f, 1.f / 65536.f,
	8.f / 65536.f, 64.f / 65536.f, 224.f / 65536.f, 448.f / 65536.f, 560.f / 65536.f, 448.f / 65536.f, 224.f / 65536.f, 64.f / 65536.f, 8.f / 65536.f,
	28.f / 65536.f, 224.f / 65536.f, 784.f / 65536.f, 1568.f / 65536.f, 1960.f / 65536.f, 1568.f / 65536.f, 784.f / 65536.f, 224.f / 65536.f, 28.f / 65536.f,
	56.f / 65536.f, 448.f / 65536.f, 1568.f / 65536.f, 3136.f / 65536.f, 3920.f / 65536.f, 3136.f / 65536.f, 1568.f / 65536.f, 448.f / 65536.f, 56.f / 65536.f,
	70.f / 65536.f, 560.f / 65536.f, 1960.f / 65536.f, 3920.f / 65536.f, 4900.f / 65536.f, 3920.f / 65536.f, 1960.f / 65536.f, 560.f / 65536.f, 70.f / 65536.f,
	56.f / 65536.f, 448.f / 65536.f, 1568.f / 65536.f, 3136.f / 65536.f, 3920.f / 65536.f, 3136.f / 65536.f, 1568.f / 65536.f, 448.f / 65536.f, 56.f / 65536.f,
	28.f / 65536.f, 224.f / 65536.f, 784.f / 65536.f, 1568.f / 65536.f, 1960.f / 65536.f, 1568.f / 65536.f, 784.f / 65536.f, 224.f / 65536.f, 28.f / 65536.f,
	8.f / 65536.f, 64.f / 65536.f, 224.f / 65536.f, 448.f / 65536.f, 560.f / 65536.f, 448.f / 65536.f, 224.f / 65536.f, 64.f / 65536.f, 8.f / 65536.f,
	1.f / 65536.f, 8.f / 65536.f, 28.f / 65536.f, 56.f / 65536.f, 70.f / 65536.f, 56.f / 65536.f, 28.f / 65536.f, 8.f / 65536.f, 1.f / 65536.f };

__global__ void kernGenerateLuminance(WindowSize window, float* color, float* albedo);
__global__ void kernRetrieveColor(WindowSize window, float* luminance, float* albedo);

// reference: https://dl.acm.org/doi/10.1145/3105762.3105770
//            http://diglib.eg.org/handle/10.2312/EGGH.HPG10.067-075
__global__ void kernBilateralFilter(
	WindowSize window, int dilation, float* denoised, float* luminance, float* normal, float* depth, float sigmaN, float sigmaD, float sigmaL);

class Denoiser {
public:
	Denoiser(Scene& Scene, PathTracer& PathTracer) :
		scene{ Scene }, window{ scene.window }, pathTracer{ PathTracer } {}
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
