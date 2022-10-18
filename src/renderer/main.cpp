#include "common.cuh"
#include "scene_loader.cuh"
#include "path_tracer.cuh"
#include "bvh.cuh"
#include "denoiser.cuh"
#include "io.cuh"
#include "gui.hpp"

#include <iomanip>
#include <stdint.h>

using namespace nagi;

int main(int argc, char* argv[]) {
	if (argc == 1) {
		std::cerr << "Error: Must specify a scene file." << std::endl;
		return EXIT_FAILURE;
	}
	std::string saveDir = ".";
	if (argc >= 3) {
		saveDir = argv[3];
		strRightStrip(saveDir, "/");
		strRightStrip(saveDir, "\\");
	}
	try {
		std::unique_ptr<SceneLoader> sceneLoader = std::make_unique<SceneLoader>(scene, argv[1]);
		sceneLoader->load();
		std::unique_ptr<BVH> bvh = std::make_unique<BVH>(scene);
		bvh->build();
		std::unique_ptr<PathTracer> pathTracer = std::make_unique<PathTracer>(scene, *bvh);
		std::unique_ptr<GUI> gui = std::make_unique<GUI>(
			"Nagi Preview Window", scene.window.width, scene.window.height, scene.config.gamma, scene.config.spp, 
			pathTracer->devFrameBuf, pathTracer->devAlbedoBuf, pathTracer->devCurrentNormalBuf, pathTracer->devCurrentDepthBuf, pathTracer->devNormalBuf, pathTracer->devDepthBuf);

		std::cout << "Start ray tracing..." << std::endl;

		while (!pathTracer->finish()) {
			float delta = pathTracer->iterate();
			if (!gui->terminated())
				gui->render(delta);
		}
		std::cout << "Ray tracing finished." << std::endl;

		auto frameBuf = pathTracer->getFrameBuffer();
		saveHDR(scene.window, frameBuf.get(), 3, saveDir + "/nagi_result_");
		savePNG(scene.window, frameBuf.get(), 3, scene.config.gamma, saveDir + "/nagi_result_");

		if (scene.config.denoiser) {
			std::unique_ptr<Denoiser> denoiser = std::make_unique<Denoiser>(scene, *pathTracer);
			auto denoisedFrameBuf = denoiser->denoise();
			saveHDR(scene.window, denoisedFrameBuf.get(), 3, saveDir + "/nagi_result_denoised_");
			savePNG(scene.window, denoisedFrameBuf.get(), 3, scene.config.gamma, saveDir + "/nagi_result_denoised_");
			denoiser.reset(nullptr);
		}

#ifdef DEB_INFO
		auto normal = pathTracer->getNormalBuffer();
		for (int i = 0; i < scene.window.pixels * 3; i++) {
			normal[i] = (normal[i] + 1.f) / 2.f;
		}
		saveHDR(scene.window, normal.get(), 3, saveDir + "/normal_");
		auto albedo = pathTracer->getAlbedoBuffer();
		saveHDR(scene.window, albedo.get(), 3, saveDir + "/albedo_");
		auto depth = pathTracer->getDepthBuffer();
		saveHDR(scene.window, depth.get(), 1, saveDir + "/depth_");
#endif // DEB_INFO

		gui.reset(nullptr);
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