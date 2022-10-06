#include "io.cuh"

#include <stb_image.h>
#include <stb_image_write.h>
#include <ctime>

namespace nagi {
std::unique_ptr<unsigned char[]> loadLDR(const std::string& filePath, int& w, int& h, int& c) {
	if (!doesFileExist(filePath)) {
		std::cerr << "Error: Image file " << filePath <<" doesn't exist." << std::endl;
		return nullptr;
	}
	std::unique_ptr<unsigned char[]> data{ stbi_load(filePath.c_str(), &w, &h, &c, STBI_default) };

	if (data == nullptr) {
		std::cerr << "Error: Can not load image file: " << filePath << std::endl;
	}
	return data;
}

std::unique_ptr<float[]> loadHDR(const std::string& filePath, int& w, int& h, int& c) {
	if (!doesFileExist(filePath)) {
		std::cerr << "Error: Image file " << filePath << " doesn't exist." << std::endl;
		return nullptr;
	}
	std::unique_ptr<float[]> data{ stbi_loadf(filePath.c_str(), &w, &h, &c, STBI_default) };

	if (data == nullptr) {
		std::cerr << "Error: Can not load image file: " << filePath << std::endl;
	}
	return data;
}

bool savePNG(const WindowSize& window, const unsigned char* buffer, const int channels, const std::string& filePath, bool timestamp) {
	time_t t = time(0);
	char timeStamp[32] = { NULL };
	if (timestamp)
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
	std::string fileName{ filePath };
	strRightStrip(fileName, ".png");
	fileName += timeStamp;
	fileName += ".png";
	if (doesFileExist(fileName)) {
		std::cerr << "Error: File " << fileName << "already exists." << std::endl;
		return false;
	}

	if (stbi_write_png(fileName.c_str(), window.width, window.height, channels, buffer, 0) == 0) {
		std::cerr << "Error: Faild to save png file " << fileName << "." << std::endl;
		return false;
	}

	return true;
}

bool saveHDR(const WindowSize& window, const float* buffer, const int channels, const std::string& filePath, bool timestamp) {
	time_t t = time(0);
	char timeStamp[32] = { NULL };
	if (timestamp)
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
	std::string fileName{ filePath };
	strRightStrip(fileName, ".hdr");
	fileName += timeStamp;
	fileName += ".hdr";
	if (doesFileExist(fileName)) {
		std::cerr << "Error: File " << fileName << "already exists." << std::endl;
		return false;
	}

	if (stbi_write_hdr(fileName.c_str(), window.width, window.height, channels, buffer) == 0) {
		std::cerr << "Error: Faild to save hdr file " << fileName << "." << std::endl;
		return false;
	}

	return true;
}
}