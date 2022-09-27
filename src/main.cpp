#include "common.cuh"
#include "scene_loader.cuh"
#include "path_tracer.cuh"
#include "user_interface.cuh"

#include <iomanip>

using namespace nagi;

int main(int argc, char* argv) {
	SceneLoader sceneLoader(scene, "res/cornell_box.json");
	PathTracer pathTracer(scene);

	try {
#ifdef NDEBUG
		sceneLoader.printDetails = false;
		pathTracer.printDetails = false;
#else
		sceneLoader.printDetails = true;
		pathTracer.printDetails = true;
#endif
		sceneLoader.load();
		pathTracer.initialize();

		//while (!pathTracer.finished()) {
		//	pathTracer.iterate();
		//}
		for (auto& trig : scene.trigBuf) {
			auto bbox = trig.bbox;
			auto min = bbox.min;
			auto max = bbox.max;
			std::cout << min.x << " " << min.y << " " << min.z << " " << std::endl;
			std::cout << max.x << " " << max.y << " " << max.z << " " << std::endl;
		}
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	std::unique_ptr<float[]> frameBuffer{ new float[PIXEL_COUNT * 3] };
	pathTracer.test1(frameBuffer.get());
	saveHDR(frameBuffer.get(), 3);
	return EXIT_SUCCESS;
}