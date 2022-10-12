#include "denoiser.cuh"
#include <OpenImageDenoise/oidn.hpp>

namespace nagi {

std::unique_ptr<float[]> Denoiser::denoise() {
	if (!devDenoised) {
		cudaMalloc((void**)&devDenoised, sizeof(float) * window.pixels * 3);
		checkCUDAError("cudaMalloc devDenoised failed.");
	}
	std::cout << "Starting denoising... ";
	std::chrono::steady_clock::time_point timer;
	timer = std::chrono::high_resolution_clock::now();

	std::unique_ptr<float[]> denoised;
	if (scene.config.denoiser == DENOISER_TYPE_FILTER)
		denoised = bilateralFilter(pathTracer.devFrameBuf, pathTracer.devAlbedoBuf, pathTracer.devNormalBuf, pathTracer.devDepthBuf);
	else if (scene.config.denoiser == DENOISER_TYPE_OIDN)
		denoised = openImageDenoiser(pathTracer.devFrameBuf, pathTracer.devAlbedoBuf, pathTracer.devNormalBuf);
	else 
		throw std::runtime_error("Error: Unknown denoiser type.");

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "  Done. Time cost: " << runningTime << " seconds." << std::endl;
	return denoised;
}

Denoiser::~Denoiser() {
	if (devDenoised) {
		cudaFree(devDenoised);
		checkCUDAError("cudaFree devDenoised failed.");
	}
}

__global__ void kernGenerateLuminance(WindowSize window, float* color, float* albedo) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;

	float a1 = albedo[idx * 3] + FLT_EPSILON;
	float a2 = albedo[idx * 3+1] + FLT_EPSILON;
	float a3 = albedo[idx * 3+2] + FLT_EPSILON;

	color[idx * 3] /= a1;
	color[idx * 3+1] /= a2;
	color[idx * 3+2] /= a3;
}
__global__ void kernRetrieveColor(WindowSize window, float* luminance, float* albedo) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;

	float a1 = albedo[idx * 3] + FLT_EPSILON;
	float a2 = albedo[idx * 3 + 1] + FLT_EPSILON;
	float a3 = albedo[idx * 3 + 2] + FLT_EPSILON;

	luminance[idx * 3] *= a1;
	luminance[idx * 3 + 1] *= a2;
	luminance[idx * 3 + 2] *= a3;
}

//todo: optimization with shared memeory
__global__ void nagi::kernBilateralFilter(
	WindowSize window, int dilation, float* denoised, float* luminance, float* normal, float* depth, float sigmaN, float sigmaZ, float sigmaL) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;

	int py = idx / window.width;
	int px = idx - py * window.width;

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
			int p = window.width * (py + i * dilation) + px + j * dilation;
			if (p == idx) {
				out += thisL;
				wSum += 1.f;
			}
			else if (p >=0 && p < window.pixels) {
				glm::vec3 n{ normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2] };
				glm::vec3 l{ luminance[p * 3], luminance[p * 3 + 1], luminance[p * 3 + 2] };
				float z = depth[p];

				float wn = powf(fmaxf(0.f, glm::dot(thisN, n)), sigmaN);
				float wz_tmp = -fabsf(thisZ - z) / sigmaZ;
				glm::vec3 w{
					__expf(-fabsf(thisL.x - l.x) / sigmaL + wz_tmp),
					__expf(-fabsf(thisL.y - l.y) / sigmaL + wz_tmp),
					__expf(-fabsf(thisL.z - l.z) / sigmaL + wz_tmp) };
				w = w * wn/* * devGaussianKernel[i]*/ / (fabsf(i * dilation) + fabsf(j * dilation));
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
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

	kernGenerateLuminance<<<blocksPerGrid, BLOCK_SIZE>>>(window, frameBuf, albedoBuf);
	checkCUDAError("kernGenerateLuminance failed.");

	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE >>>(window, 1, devDenoised, frameBuf, normalBuf, depthBuf, 64.f, 1.f, 4.f);
	checkCUDAError("kernBilateralFilter failed.");
	std::swap(devDenoised, frameBuf);
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(window, 2, devDenoised, frameBuf, normalBuf, depthBuf, 64.f, 1.f, 4.f);
	checkCUDAError("kernBilateralFilter failed.");
	std::swap(devDenoised, frameBuf);
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(window, 3, devDenoised, frameBuf, normalBuf, depthBuf, 64.f, 1.f, 4.f);
	checkCUDAError("kernBilateralFilter failed.");

	kernRetrieveColor<<<blocksPerGrid, BLOCK_SIZE >>>(window, devDenoised, albedoBuf);
	checkCUDAError("kernRetrieveColor failed.");

	std::unique_ptr<float[]> denoised{ new float[window.pixels * 3] };
	cudaMemcpy(denoised.get(), devDenoised, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devDenoised failed.");

	return denoised;
}

std::unique_ptr<float[]>  Denoiser::openImageDenoiser(float* devFrameBuf, float* devAlbedoBuf, float* devNormalBuf) {
	// oidn only supports cpu
	std::unique_ptr<float[]> frameBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> albedoBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> normalBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> denoisedBuf{ new float[window.pixels * 3] };
	cudaMemcpy(frameBuf.get(), devFrameBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devFrameBuf failed.");
	cudaMemcpy(albedoBuf.get(), devAlbedoBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devAlbedoBuf failed.");
	cudaMemcpy(normalBuf.get(), devNormalBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devNormalBuf failed.");

	auto device = oidn::newDevice();
	device.commit();

	// Create a filter for denoising a beauty (color) image using optional auxiliary images too
	auto filter = device.newFilter("RT"); // generic ray tracing filter
	filter.setImage("color", frameBuf.get(), oidn::Format::Float3, window.width, window.height); // beauty
	filter.setImage("albedo", albedoBuf.get(), oidn::Format::Float3, window.width, window.height); // auxiliary
	filter.setImage("normal", normalBuf.get(), oidn::Format::Float3, window.width, window.height); // auxiliary
	filter.setImage("output", denoisedBuf.get(), oidn::Format::Float3, window.width, window.height); // denoised beauty
	filter.set("hdr", true); // beauty image is HDR
	filter.commit();

	filter.execute();
	const char* errorMessage;
	if (device.getError(errorMessage) != oidn::Error::None)
		throw std::runtime_error(errorMessage);

	return denoisedBuf;
}

}