#include "common.cuh"
#include "scene_loader.cuh"
#include "path_tracer.cuh"
#include "user_interface.cuh"
#include "denoiser.cuh"

#include <iomanip>

#include <OpenImageDenoise/oidn.hpp>

using namespace nagi;

int main(int argc, char* argv[]) {
	SceneLoader sceneLoader(scene, argv[1]);
	PathTracer pathTracer(scene);
	try {
		sceneLoader.printDetails = true;
		pathTracer.printDetails = true;
		sceneLoader.load();
		pathTracer.initialize();
		pathTracer.iterate();
		auto frameBuf = pathTracer.getFrameBuffer();
		saveHDR(scene.config.window, frameBuf.get(), 3);

		auto normal = pathTracer.getNormalBuffer();
		saveHDR(scene.config.window, normal.get(), 3, "./normal_");
		auto albedo = pathTracer.getAlbedoBuffer();
		saveHDR(scene.config.window, albedo.get(), 3, "./albedo_");

		if (scene.config.denoiser != 0) {
			Denoiser denoiser(scene, pathTracer);
			auto denoisedFrameBuf = denoiser.denoise();
			saveHDR(scene.config.window, denoisedFrameBuf.get(), 3, "./nagi_result_denoised_");
		}
		//std::unique_ptr<float[]> denoisedBuf{ new float[PIXEL_COUNT * 3] };

		//pathTracer.copyFrameBuffer(denoisedBuf.get());
		//saveHDR(denoisedBuf.get(), 3, "./nagi_result_denoised_bi_");

		//auto device = oidn::newDevice();
		//device.commit();
		////OIDNBuffer beauty = oidnNewSharedBuffer(defaultDeivce, frameBuf, sizeof(float) * PIXEL_COUNT * 3);
		////OIDNBuffer albedo = oidnNewSharedBuffer(defaultDeivce, albedoBuf, sizeof(float) * PIXEL_COUNT * 3);
		////OIDNBuffer normal = oidnNewSharedBuffer(defaultDeivce, normalBuf, sizeof(float) * PIXEL_COUNT * 3);

		//auto albedoBuf = pathTracer.getAlbedoBuffer();
		//auto normalBuf = pathTracer.getNormalBuffer();
		//// Create a filter for denoising a beauty (color) image using optional auxiliary images too
		//auto filter = device.newFilter("RT"); // generic ray tracing filter
		//filter.setImage("color", frameBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // beauty
		//filter.setImage("albedo", albedoBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // auxiliary
		//filter.setImage("normal", normalBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // auxiliary
		//filter.setImage("output", denoisedBuf.get(), oidn::Format::Float3, WINDOW_WIDTH, WINDOW_HEIGHT); // denoised beauty
		//filter.set("hdr", true); // beauty image is HDR
		//filter.commit();

		//filter.execute();
		//const char* errorMessage;
		//if (device.getError(errorMessage) != oidn::Error::None)
		//	std::cerr<<errorMessage<<std::endl;

		//saveHDR(denoisedBuf.get(), 3, "./nagi_result_denoised_");

	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}