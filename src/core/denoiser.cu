#include "denoiser.cuh"
#include <OpenImageDenoise/oidn.hpp>

namespace nagi {

std::unique_ptr<float[]> Denoiser::denoise() {
	if (!devDenoised) {
		cudaRun(cudaMalloc((void**)&devDenoised, sizeof(float) * window.pixels * 3));
	}
	std::cout << "Starting denoising... ";
	std::chrono::steady_clock::time_point timer;
	timer = std::chrono::high_resolution_clock::now();

	// oidn only supports cpu
	std::unique_ptr<float[]> frameBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> albedoBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> normalBuf{ new float[window.pixels * 3] };
	std::unique_ptr<float[]> denoisedBuf{ new float[window.pixels * 3] };
	cudaRun(cudaMemcpy(frameBuf.get(), pathTracer.devFrameBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost));
	cudaRun(cudaMemcpy(albedoBuf.get(), pathTracer.devAlbedoBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost));
	cudaRun(cudaMemcpy(normalBuf.get(), pathTracer.devNormalBuf, sizeof(float) * window.pixels * 3, cudaMemcpyDeviceToHost));

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

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "  Done. Time cost: " << runningTime << " seconds." << std::endl;
	return denoisedBuf;
}

Denoiser::~Denoiser() {
	if (devDenoised) {
		cudaRun(cudaFree(devDenoised));
	}
}

}