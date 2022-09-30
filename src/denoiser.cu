#include "denoiser.cuh"
#include <OpenImageDenoise/oidn.hpp>

namespace nagi {

std::unique_ptr<float[]> Denoiser::denoise() {
	if (!devDenoised) {
		cudaMalloc((void**)&devDenoised, sizeof(float) * PIXEL_COUNT * 3);
		checkCUDAError("cudaMalloc devDenoised failed.");
	}
	std::cout << "Starting denoising... ";
	std::chrono::steady_clock::time_point timer;
	timer = std::chrono::high_resolution_clock::now();

	std::unique_ptr<float[]> denoised;
	if (scene.config.denoiser == 1) 
		denoised = bilateralFilter(pathTracer.devFrameBuf, pathTracer.devAlbedoBuf, pathTracer.devNormalBuf, pathTracer.devDepthBuf);
	else if (scene.config.denoiser == 2) 
		denoised = openImageDenoiser(pathTracer.devFrameBuf, pathTracer.devAlbedoBuf, pathTracer.devNormalBuf);
	else 
		throw std::runtime_error("Error: Unknown denoiser type. Allowing types are: 1 (Bilateral Filtering); 2 (Open Image Denoise).");

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "  Done. Time cost: " << runningTime << " seconds." << std::endl;
	return denoised;
}

Denoiser::~Denoiser() {
	if (devDenoised) {
		cudaFree(devDenoised);
		checkCUDAError2("cudaFree devDenoised failed.");
	}
}

__global__ void kernGenerateLuminance(float* color, float* albedo) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;

	float a1 = albedo[idx * 3] + FLT_EPSILON;
	float a2 = albedo[idx * 3+1] + FLT_EPSILON;
	float a3 = albedo[idx * 3+2] + FLT_EPSILON;

	color[idx * 3] /= a1;
	color[idx * 3+1] /= a2;
	color[idx * 3+2] /= a3;
}
__global__ void kernRetrieveColor(float* luminance, float* albedo) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;

	float a1 = albedo[idx * 3] + FLT_EPSILON;
	float a2 = albedo[idx * 3 + 1] + FLT_EPSILON;
	float a3 = albedo[idx * 3 + 2] + FLT_EPSILON;

	luminance[idx * 3] *= a1;
	luminance[idx * 3 + 1] *= a2;
	luminance[idx * 3 + 2] *= a3;
}

//todo: optimization with shared memeory
__global__ void nagi::kernBilateralFilter(
	float* denoised, float* luminance, float* normal, float* depth, float sigmaN, float sigmaZ, float sigmaL) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;

	int py = idx / WINDOW_WIDTH;
	int px = idx - py * WINDOW_HEIGHT;

	int pixels[FILTER_SIZE_SQUARE];

	glm::vec3 thisL{ luminance[idx*3], luminance[idx*3 + 1], luminance[idx*3 + 2] };
	glm::vec3 thisN{ normal[idx*3], normal[idx*3 + 1], normal[idx*3 + 2] };
	float thisZ{ depth[idx] };
	glm::vec3 out{ 0.f };
	glm::vec3 wSum{ 0.f };


#pragma unroll
	for (int i = -FILTER_SIZE_HALF; i <= FILTER_SIZE_HALF; i++) {
#pragma unroll
		for (int j = -FILTER_SIZE_HALF; j <= FILTER_SIZE_HALF; j++) {
			int p = WINDOW_WIDTH * (py + i) + px + j;
			if (p == idx) {
				out += thisL;
				wSum += 1.f;
			}
			else if (p >=0 && p < PIXEL_COUNT) {
				glm::vec3 n{ normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2] };
				glm::vec3 l{ luminance[p * 3], luminance[p * 3 + 1], luminance[p * 3 + 2] };
				float z = depth[p];

				float wn = powf(fmaxf(0.f, glm::dot(thisN, n)), sigmaN);
				float wz_tmp = -fabsf(thisZ - z) / sigmaZ;
				glm::vec3 w{
					__expf(-fabsf(thisL.x - l.x) / sigmaL + wz_tmp),
					__expf(-fabsf(thisL.y - l.y) / sigmaL + wz_tmp),
					__expf(-fabsf(thisL.z - l.z) / sigmaL + wz_tmp) };
				w = w * wn/* * devGaussianKernel[i]*/ / (fabsf(i) + fabsf(j));
				out += w * l;
				wSum += w;
			}
		}
	}
	out /= wSum;
	denoised[idx*3] = out.x;
	denoised[idx*3+1] = out.y;
	denoised[idx*3+2] = out.z;
}

// todo: bilateral filter will introduce aliasing at edges (as they appear in gbuffers). still don't know how to solve it.
std::unique_ptr<float[]> Denoiser::bilateralFilter(float* frameBuf, float* albedoBuf, float* normalBuf, float* depthBuf) {
	dim3 blocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);

	kernGenerateLuminance<<<blocksPerGrid, BLOCK_SIZE>>>(frameBuf, albedoBuf);
	checkCUDAError("kernGenerateLuminance failed.");

	kernBilateralFilter << <blocksPerGrid, BLOCK_SIZE >> > (devDenoised, frameBuf, normalBuf, depthBuf, 64.f, 1.f, 4.f);
	checkCUDAError("kernBilateralFilter failed.");

	kernRetrieveColor << <blocksPerGrid, BLOCK_SIZE >> > (devDenoised, albedoBuf);
	checkCUDAError("kernRetrieveColor failed.");

	std::unique_ptr<float[]> denoised{ new float[PIXEL_COUNT * 3] };
	cudaMemcpy(denoised.get(), devDenoised, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devDenoised failed.");

	return denoised;
}

std::unique_ptr<float[]>  Denoiser::openImageDenoiser(float* devFrameBuf, float* devAlbedoBuf, float* devNormalBuf) {
	// oidn only supports cpu
	std::unique_ptr<float[]> frameBuf{ new float[PIXEL_COUNT * 3] };
	std::unique_ptr<float[]> albedoBuf{ new float[PIXEL_COUNT * 3] };
	std::unique_ptr<float[]> normalBuf{ new float[PIXEL_COUNT * 3] };
	std::unique_ptr<float[]> denoisedBuf{ new float[PIXEL_COUNT * 3] };
	cudaMemcpy(frameBuf.get(), devFrameBuf, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devFrameBuf failed.");
	cudaMemcpy(albedoBuf.get(), devAlbedoBuf, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devAlbedoBuf failed.");
	cudaMemcpy(normalBuf.get(), devNormalBuf, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devNormalBuf failed.");

	auto device = oidn::newDevice();
	device.commit();

	// Create a filter for denoising a beauty (color) image using optional auxiliary images too
	auto filter = device.newFilter("RT"); // generic ray tracing filter
	filter.setImage("color", frameBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // beauty
	filter.setImage("albedo", albedoBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // auxiliary
	filter.setImage("normal", normalBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // auxiliary
	filter.setImage("output", denoisedBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // denoised beauty
	filter.set("hdr", true); // beauty image is HDR
	filter.commit();

	filter.execute();
	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
		throw std::runtime_error(errorMessage);

	return denoisedBuf;
}

}