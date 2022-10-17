#ifndef IO_CUH
#define IO_CUH

#include "common.cuh"

#include <stb_image.h>
#include <stb_image_write.h>

namespace nagi {

inline std::unique_ptr<unsigned char[]> loadLDR(const std::string& filePath, int& w, int& h, int& c) {
	if (!doesFileExist(filePath)) {
		throw std::runtime_error("Error: Image file doesn't exist.");
	}
	std::unique_ptr<unsigned char[]> data{ stbi_load(filePath.c_str(), &w, &h, &c, STBI_default) };

	if (data == nullptr) {
		std::cerr << "Error: Can not load image file: " << filePath << std::endl;
	}
	return data;
}

inline std::unique_ptr<float[]> loadHDR(const std::string& filePath, int& w, int& h, int& c) {
	if (!doesFileExist(filePath)) {
		throw std::runtime_error("Error: Image file doesn't exist.");
	}
	std::unique_ptr<float[]> data{ stbi_loadf(filePath.c_str(), &w, &h, &c, STBI_default) };

	if (data == nullptr) {
		throw std::runtime_error("Error: Can not load image file.");
	}
	return data;
}

inline void savePNG(
	const WindowSize& window, const unsigned char* buffer, int channels, const std::string& filePath = "./nagi_result_", bool timestamp = true) {
	time_t t = time(0);
	char timeStamp[32] = { NULL };
	if (timestamp)
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
	std::string fileName{ filePath };
	strRightStrip(fileName, ".png");
	fileName += timeStamp;
	fileName += ".png";
	if (doesFileExist(fileName)) {
		throw std::runtime_error("Error: File already exists.");
	}

	if (stbi_write_png(fileName.c_str(), window.width, window.height, channels, buffer, 0) == 0) {
		throw std::runtime_error("Error: Faild to save png file.");
	}
}

inline void savePNG(
	const WindowSize& window, const float* buffer, int channels, const float gamma = 2.2, const std::string& filePath = "./nagi_result_", bool timestamp = true) {
	std::unique_ptr<unsigned char[]> buf{ new unsigned char[window.pixels * channels] };
	for (int i = 0; i < scene.window.pixels * channels; i++) {
		buf[i] = glm::clamp((int)(powf(buffer[i], 1.f / gamma) * 255.f), 0, 255);
	}
	savePNG(window, buf.get(), channels, filePath, timestamp);
}

inline void saveHDR(
	const WindowSize& window, const float* buffer, const int channels, const std::string& filePath = "./nagi_result_", bool timestamp = true) {
	time_t t = time(0);
	char timeStamp[32] = { NULL };
	if (timestamp)
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
	std::string fileName{ filePath };
	strRightStrip(fileName, ".hdr");
	fileName += timeStamp;
	fileName += ".hdr";
	if (doesFileExist(fileName)) {
		throw std::runtime_error("Error: File already exists.");
	}

	if (stbi_write_hdr(fileName.c_str(), window.width, window.height, channels, buffer) == 0) {
		throw std::runtime_error("Error: Faild to save hdr file.");
	}
}

}

#endif // !USER_INTERFACE_CUH
