#ifndef GUI_HPP

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#define IM_VEC2_CLASS_EXTRA constexpr ImVec2(const glm::vec2& f) : x(f.x), y(f.y) {} operator glm::vec2() const { return glm::vec2(x,y); }
#define IM_VEC4_CLASS_EXTRA constexpr ImVec4(const glm::vec4& f) : x(f.x), y(f.y), z(f.z), w(f.w) {} operator glm::vec4() const { return glm::vec4(x,y,z,w); }
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <cstdio>
#include <chrono>
#include <memory>
#include <exception>
#include <string>
#include <vector>
#include <list>

#define BLOCK_SIZE 128

inline void glfw_error_callback(int error, const char* description) {
	std::string error_msg = "GLFW Error: ";
	error_msg += error;
	error_msg += description;
    throw std::runtime_error(error_msg);
}

template <typename T>
inline T listSum(const std::list<T> s) {
	T sum;
	for (auto& it = s.begin(); it != s.end(); it++) {
		sum += *it;
	}
	return sum;
}

#define FILTER_SIZE 5
#define FILTER_SIZE_HALF 2
#define FILTER_AREA 25
#define FILTER_AREA_HALF 12

// CUDA 11.3 has added device code support for new C++ keywords: `constexpr` and `auto`.
// In CUDA C++, `__device__` and `__constant__` variables can now be declared `constexpr`.
// The constexpr variables can be used in constant expressions, where they are evaluated at
// compile time, or as normal variables in contexts where constant expressions are not required.

__device__ constexpr float devGaussianKernel[25]{
	1.f/* / 273.f*/, 4.f /* / 273.f*/, 7.f /* / 273.f*/, 4.f /* / 273.f*/, 1.f/* / 273.f*/,
	4.f/* / 273.f*/, 16.f/* / 273.f*/, 26.f/* / 273.f*/, 16.f/* / 273.f*/, 4.f/* / 273.f*/,
	7.f/* / 273.f*/, 26.f/* / 273.f*/, 41.f/* / 273.f*/, 26.f/* / 273.f*/, 7.f/* / 273.f*/,
	4.f/* / 273.f*/, 16.f/* / 273.f*/, 26.f/* / 273.f*/, 16.f/* / 273.f*/, 4.f/* / 273.f*/,
	1.f/* / 273.f*/, 4.f /* / 273.f*/, 7.f /* / 273.f*/, 4.f /* / 273.f*/, 1.f/* / 273.f*/
};

namespace nagi {
    
class GUI {
public:
	GUI(const std::string& windowName, int w, int h, float gamma, int spp,
		float* devResult, float* devAlbedo, float* devNormal, float* devDepth, float* devFinalNormal, float* devFinalDepth);
	~GUI() { terminate(); }
	GUI(const GUI&) = delete;
	void operator=(const GUI&) = delete;

	bool terminated() const { 
		return window == nullptr;
	}
	void terminate();
	void denoise();
	void render(float delta);
	void copyToFrameBuffer();

	glm::vec4 clearColor{ 0.45f, 0.55f, 0.60f, 1.00f };

	GLFWwindow* window = nullptr;
	int width, height, pixels;
	int step{ 1 };
	int totalSpp;
	float gamma;
	int present{ 0 };
	float sigmaN{ 64.f }, sigmaZ{ 1.f }, sigmaL{ 4.f };
	bool denoiser{ false };
	GLuint pbo;
	cudaGraphicsResource* regesitered_pbo;
	uchar4* devFrameBuffer{ nullptr };
	float* devDenoisedResult1{ nullptr };
	float* devDenoisedResult2{ nullptr };
	float* devResult{ nullptr };
	float* devNormal{ nullptr };
	float* devAlbedo{ nullptr };
	float* devDepth{ nullptr };

	float* devFinalNormal{ nullptr };
	float* devFinalDepth{ nullptr };
};

}

#endif // !GUI_HPP
