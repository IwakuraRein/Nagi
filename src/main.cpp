#include "common.cuh"
#include "scene_loader.cuh"
#include "path_tracer.cuh"
#include "user_interface.cuh"

#include <iomanip>

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
		std::unique_ptr<float[]> frameBuffer = pathTracer.getFrameBuffer();
		saveHDR(frameBuffer.get(), 3);
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}