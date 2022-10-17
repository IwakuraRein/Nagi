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
#include <memory>
#include <exception>
#include <string>

#define BLOCK_SIZE 128

inline void glfw_error_callback(int error, const char* description)
{
	std::string error_msg = "Glfw Error ";
	error_msg += error;
	error_msg += description;
    throw std::runtime_error(error_msg);
}

int imGuiDemo();

#define FILTER_SIZE 9
#define FILTER_SIZE_HALF 4
#define FILTER_AREA 81
#define FILTER_AREA_HALF 40

// CUDA 11.3 has added device code support for new C++ keywords: `constexpr` and `auto`.
// In CUDA C++, `__device__` and `__constant__` variables can now be declared `constexpr`.
// The constexpr variables can be used in constant expressions, where they are evaluated at
// compile time, or as normal variables in contexts where constant expressions are not required.

__device__ constexpr float devGaussianKernel[81]{
	1.f / 65536.f, 8.f / 65536.f, 28.f / 65536.f, 56.f / 65536.f, 70.f / 65536.f, 56.f / 65536.f, 28.f / 65536.f, 8.f / 65536.f, 1.f / 65536.f,
	8.f / 65536.f, 64.f / 65536.f, 224.f / 65536.f, 448.f / 65536.f, 560.f / 65536.f, 448.f / 65536.f, 224.f / 65536.f, 64.f / 65536.f, 8.f / 65536.f,
	28.f / 65536.f, 224.f / 65536.f, 784.f / 65536.f, 1568.f / 65536.f, 1960.f / 65536.f, 1568.f / 65536.f, 784.f / 65536.f, 224.f / 65536.f, 28.f / 65536.f,
	56.f / 65536.f, 448.f / 65536.f, 1568.f / 65536.f, 3136.f / 65536.f, 3920.f / 65536.f, 3136.f / 65536.f, 1568.f / 65536.f, 448.f / 65536.f, 56.f / 65536.f,
	70.f / 65536.f, 560.f / 65536.f, 1960.f / 65536.f, 3920.f / 65536.f, 4900.f / 65536.f, 3920.f / 65536.f, 1960.f / 65536.f, 560.f / 65536.f, 70.f / 65536.f,
	56.f / 65536.f, 448.f / 65536.f, 1568.f / 65536.f, 3136.f / 65536.f, 3920.f / 65536.f, 3136.f / 65536.f, 1568.f / 65536.f, 448.f / 65536.f, 56.f / 65536.f,
	28.f / 65536.f, 224.f / 65536.f, 784.f / 65536.f, 1568.f / 65536.f, 1960.f / 65536.f, 1568.f / 65536.f, 784.f / 65536.f, 224.f / 65536.f, 28.f / 65536.f,
	8.f / 65536.f, 64.f / 65536.f, 224.f / 65536.f, 448.f / 65536.f, 560.f / 65536.f, 448.f / 65536.f, 224.f / 65536.f, 64.f / 65536.f, 8.f / 65536.f,
	1.f / 65536.f, 8.f / 65536.f, 28.f / 65536.f, 56.f / 65536.f, 70.f / 65536.f, 56.f / 65536.f, 28.f / 65536.f, 8.f / 65536.f, 1.f / 65536.f };

namespace nagi {
    
class GUI {
public:
	GUI(const std::string& windowName, int w, int h, float gamma, float* devResultBuffer, float* devNormalBuffer, float* devAlbedoBuffer, float* devDepthBuffer);
	~GUI() { terminate(); }
	GUI(const GUI&) = delete;
	void operator=(const GUI&) = delete;

	bool terminated() const { 
		return window == nullptr;
	}
	void terminate();
	void denoise() {}
	void render();
	void copyToFrameBuffer();

	glm::vec4 clearColor{ 0.45f, 0.55f, 0.60f, 1.00f };

	GLFWwindow* window = nullptr;
	int width, height, pixels;
	unsigned int counter{ 1 };
	float gamma;
	GLuint pbo;
	cudaGraphicsResource* regesitered_pbo;
	uchar4* devFrameBuffer{ nullptr };
	float* devResultBuffer{ nullptr };
	float* devNormalBuffer{ nullptr };
	float* devAlbedoBuffer{ nullptr };
	float* devDepthBuffer{ nullptr };
};

}

#endif // !GUI_HPP
