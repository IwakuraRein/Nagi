#ifndef USER_INTERFACE_HPP
#define USER_INTERFACE_HPP

#include "common.cuh"

namespace nagi {

	bool savePNG(const unsigned char* buffer, int channels = 4, const std::string& folderPath = "./");
	bool saveHDR(const float* buffer, const int channels, const std::string& folderPath = "./");
}

#endif // !USER_INTERFACE_HPP
