#include "common.cuh"
#include "scene_loader.cuh"
#include "path_tracer.cuh"
#include "bvh.cuh"
#include "denoiser.cuh"
#include "io.cuh"

#include <iomanip>

#include <OpenImageDenoise/oidn.hpp>

using namespace nagi;

int main(int argc, char* argv[]) {
	if (argc == 1) {
		std::cerr << "Error: Must specify a scene file." << std::endl;
		return EXIT_FAILURE;
	}
	try {

		std::unique_ptr<SceneLoader> sceneLoader = std::make_unique<SceneLoader>(scene, argv[1]);
		sceneLoader->printDetails = true;
		sceneLoader->load();
		std::unique_ptr<BVH> bvh = std::make_unique<BVH>(scene);
		bvh->build();
		std::unique_ptr<PathTracer> pathTracer = std::make_unique<PathTracer>(scene, *bvh);
		pathTracer->printDetails = true;

		pathTracer->initialize();
		pathTracer->iterate();
		auto frameBuf = pathTracer->getFrameBuffer();
		saveHDR(scene.config.window, frameBuf.get(), 3);

		if (scene.config.denoiser != DENOISER_TYPE_NONE) {
			std::unique_ptr<Denoiser> denoiser = std::make_unique<Denoiser>(scene, *pathTracer);
			auto denoisedFrameBuf = denoiser->denoise();
			saveHDR(scene.config.window, denoisedFrameBuf.get(), 3, "./nagi_result_denoised_");
			denoiser.reset(nullptr);
		}

		//auto normal = pathTracer->getNormalBuffer();
		//saveHDR(scene.config.window, normal.get(), 3, "./normal_");
		auto albedo = pathTracer->getAlbedoBuffer();
		saveHDR(scene.config.window, albedo.get(), 3, "./albedo_");
		//auto depth = pathTracer->getDepthBuffer();
		//saveHDR(scene.config.window, depth.get(), 1, "./depth_");

		pathTracer.reset(nullptr);
		bvh.reset(nullptr);
		sceneLoader.reset(nullptr);
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}