#ifndef COMMON_HPP
#define COMMON_HPP

#define GLM_COORDINATE_SYSTEM GLM_RIGHT_HANDED
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <map>
#include <limits>
#include <exception>

// predefined values
#define PI 3.14159265358979323846264338327950288f
#define HALF_PI 1.57079632679489661923132169163975144f
#define QUAT_PI 0.785398163397448309615660845819875721f
#define INV_PI 0.318309886183790671537767526745028724f

// CUDA 11.3 has added device code support for new C++ keywords: `constexpr` and `auto`.
// In CUDA C++, `__device__` and `__constant__` variables can now be declared `constexpr`.
// The constexpr variables can be used in constant expressions, where they are evaluated at
// compile time, or as normal variables in contexts where constant expressions are not required.
constexpr int BLOCK_SIZE = 128;
constexpr int WINDOW_WIDTH = 1024;
constexpr int WINDOW_HEIGHT = 1024;
constexpr int PIXEL_COUNT = WINDOW_WIDTH * WINDOW_HEIGHT;
constexpr float INV_WIDTH = 1.f / WINDOW_WIDTH;
constexpr float INV_HEIGHT = 1.f / WINDOW_HEIGHT;

// help functions
inline void checkCUDAError(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::string errMsg{"Cuda error: "};
		errMsg += msg;
		errMsg += cudaGetErrorString(err);
		throw std::runtime_error(errMsg);
	}
}
inline void checkCUDAError2(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		std::string errMsg{"Cuda error: "};
		errMsg += msg;
		errMsg += cudaGetErrorString(err);
		std::cerr << errMsg << std::endl;
	}
}

inline __host__ __device__ unsigned int hash(unsigned int a) {
	a = (a + 0x7ed55d16) + (a << 12);
	a = (a ^ 0xc761c23c) ^ (a >> 19);
	a = (a + 0x165667b1) + (a << 5);
	a = (a + 0xd3a2646c) ^ (a << 9);
	a = (a + 0xfd7046c5) + (a << 3);
	a = (a ^ 0xb55a4f09) ^ (a >> 16);
	return a;
};

template <typename T, typename... Rest>
inline void hashCombine(std::size_t& seed, const T& v, Rest... rest) {
	std::hash<T> hasher;
	seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
	hashCombine(seed, rest...);
}

bool doesFileExist(const std::string& name);

inline bool strEndWith(const std::string& str, const std::string& tail) {
	return str.compare(str.size() - tail.size(), tail.size(), tail) == 0;
}

inline bool strStartWith(const std::string& str, const std::string& head) {
	return str.compare(0, head.size(), head) == 0;
}

std::string getFileName(const std::string& str);

std::string strLeftStrip(const std::string& str, const std::string& strip);

std::string strRightStrip(const std::string& str, const std::string& strip);

inline int pixel2Dto1D(int x, int y) {
	return y * WINDOW_WIDTH + x;
}

#ifndef hasItem(X, Y)
#define hasItem(X, Y) (X.find(Y) != X.end())
#endif


// common data structures
namespace nagi {

struct Transform {
	glm::vec3 position{ 0.f, 0.f, 0.f };
	glm::vec3 rotation{ 0.f, 0.f, 0.f };
	glm::vec3 scale{ 1.f, 1.f, 1.f };
	glm::mat4 transformMat;
	glm::mat4 invTransformMat;
};

__device__ __host__ glm::mat4 getTransformMat(
	const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale);
__device__ __host__ void updateTransformMat(Transform* t);
__device__ __host__ glm::mat4 getRotationMat(const glm::vec3& rotation);
__device__ __host__ void vecTransform(glm::vec3* vec, const glm::mat4& mat, float T = 1.f);
__device__ __host__ void vecTransform2(glm::vec3* vec, const glm::mat4& mat, float T = 1.f);

struct Camera {
	float near{ 0.01f };
	float far{ 1000.f };
	float fov{ 60.f };
	float aspect{ WINDOW_WIDTH / WINDOW_HEIGHT };
	glm::vec3 position{ 0.f, 0.f, 0.f };
	glm::vec3 upDir{ 0.f, -1.f, 0.f };
	glm::vec3 rightDir{ 1.f, 0.f, 0.f };
	glm::vec3 forwardDir{ 0.f, 0.f, 1.f };

	glm::vec3 screenOrigin;
	float pixelWidth, pixelHeight, halfPixelWidth, halfPixelHeight;
};

struct BoundingBox { // AABB
	glm::vec3 min;
	glm::vec3 max;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

struct Triangle {
	Vertex vert0;
	Vertex vert1;
	Vertex vert2;
	BoundingBox bbox;
};

struct Ray {
	glm::vec3 origin;
	glm::vec3 dir;
	glm::vec3 invDir;
};

struct Path {
	Ray ray;
	glm::vec3 color;
	int remainingBounces;
	int pixelIdx;
};

struct Object {
	Transform transform;
	int trigIdxStart;
	int trigIdxEnd;
	int mtlIdx;
	BoundingBox bbox;
};

struct Material {
	bool isLightSource;
	glm::vec3 albedo;
	float emission;
	float transparent;
	float roughness;
	float metalness;
	float ior;
};

struct Configuration {
	float spp;
	float maxBounce;
	float alpha;
	float gamma;
};

struct Scene {
	Configuration config;

	Camera cam;

	BoundingBox bbox;

	std::vector<Object> objBuf;
	std::vector<Material> mtlBuf;
	std::vector<Triangle> trigBuf;

	//Object* devObjBuf;
	//Material* devMtlBuf;
	//Triangle* devTrigBuf;

	//thrust::device_vector<Object> devObjBuf2;
	//thrust::device_vector<Material> devMtlBuf2;
	//thrust::device_vector<Triangle> devTrigBuf2;

	// The frame buffer is a rectangle of pixels stored from left-to-right, top-to-bottom.
	//float* devFrameBuf;
	//thrust::device_vector<float> devFrameBuf2;
};
extern Scene scene; // global variable

}

#endif // !COMMON_HPP
