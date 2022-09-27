#include "user_interface.cuh"

#include <stb_image_write.h>

#include <ctime>

namespace nagi {

	bool savePNG(const unsigned char* buffer, const int channels, const std::string& folderPath) {
		time_t t = time(0);
		char timeStamp[32] = { NULL };
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
		std::string fileName{ folderPath + "result_" };
		fileName += timeStamp;
		fileName += ".png";
		if (doesFileExist(fileName)) {
			std::cerr << "Error: File " << fileName << "already exists." << std::endl;
			return false;
		}

		if (stbi_write_png(fileName.c_str(), WINDOW_WIDTH, WINDOW_HEIGHT, channels, buffer, 0) == 0) {
			std::cerr << "Error: Faild to save screen shot " << fileName << "." << std::endl;
			return false;
		}

		return true;
	}

	bool saveHDR(const float* buffer, const int channels, const std::string& folderPath) {
		time_t t = time(0);
		char timeStamp[32] = { NULL };
		strftime(timeStamp, sizeof(timeStamp), "%Y-%m-%d_%H-%M-%S", localtime(&t));
		std::string fileName{ folderPath + "nagi_result_" };
		fileName += timeStamp;
		fileName += ".hdr";
		if (doesFileExist(fileName)) {
			std::cerr << "Error: File " << fileName << "already exists." << std::endl;
			return false;
		}

		if (stbi_write_hdr(fileName.c_str(), WINDOW_WIDTH, WINDOW_HEIGHT, channels, buffer) == 0) {
			std::cerr << "Error: Faild to save screen shot " << fileName << "." << std::endl;
			return false;
		}

		return true;
	}
}