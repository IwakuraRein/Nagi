#ifndef IO_CUH
#define IO_CUH

#include "common.cuh"

namespace nagi {

std::unique_ptr<unsigned char[]> loadLDR(const std::string& filePath, int& width, int& height, int& channels);
std::unique_ptr<float[]> loadHDR(const std::string& filePath, int& width, int& height, int& channels);
bool savePNG(
	const WindowSize& window, const unsigned char* buffer, int channels, const std::string& filePath = "./nagi_result_", bool timestamp = true);
bool saveHDR(
	const WindowSize& window, const float* buffer, const int channels, const std::string& filePath = "./nagi_result_", bool timestamp = true);

}

#endif // !USER_INTERFACE_CUH
