#ifndef USER_INTERFACE_CUH
#define USER_INTERFACE_CUH

#include "common.cuh"

namespace nagi {

bool savePNG(const unsigned char* buffer, int channels = 4, const std::string& filePath = "./nagi_result_", bool timestamp = true);
bool saveHDR(const float* buffer, const int channels, const std::string& filePath = "./nagi_result_", bool timestamp = true);

}

#endif // !USER_INTERFACE_CUH
