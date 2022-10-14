#ifndef COMMON_CUH
#define COMMON_CUH

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
#include <list>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <limits>
#include <exception>

// predefined values
#define PI          3.14159265358979323846264338327950288f
#define TWO_PI      6.28318530717958647692528676655900576f
#define HALF_PI     1.57079632679489661923132169163975144f
#define QUAT_PI     0.785398163397448309615660845819875721f
#define INV_PI      0.318309886183790671537767526745028724f
#define INV_TWO_PI  0.159154943091895335768883763372514362f

#define MTL_TYPE_LAMBERT 0
#define MTL_TYPE_MICROFACET 1
#define MTL_TYPE_GLASS 2
#define MTL_TYPE_SPECULAR 3
#define MTL_TYPE_LIGHT_SOURCE 4

#define PIXEL_TYPE_DIFFUSE  0
#define PIXEL_TYPE_GLOSSY   1
#define PIXEL_TYPE_SPECULAR 2

#define DENOISER_TYPE_NONE 0
#define DENOISER_TYPE_FILTER 1
#define DENOISER_TYPE_OIDN 2

#define TEXTURE_TYPE_BASE 0
#define TEXTURE_TYPE_ROUGHNESS 1
#define TEXTURE_TYPE_METALLIC 2
#define TEXTURE_TYPE_NORMAL 3
#define TEXTURE_TYPE_OCCLUSION 4

#define BLOCK_SIZE 128

#define HALF_FILM_HEIGHT 0.012f // 24 x 36 mm film
#define CAMERA_MULTIPLIER 1000.f // multiply a large number to work around float's precision problem

// CUDA 11.3 has added device code support for new C++ keywords: `constexpr` and `auto`.
// In CUDA C++, `__device__` and `__constant__` variables can now be declared `constexpr`.
// The constexpr variables can be used in constant expressions, where they are evaluated at
// compile time, or as normal variables in contexts where constant expressions are not required.
//constexpr int BLOCK_SIZE = 128;
//constexpr int WINDOW_WIDTH = 1024;
//constexpr int WINDOW_HEIGHT = 1024;
//constexpr int PIXEL_COUNT = WINDOW_WIDTH * WINDOW_HEIGHT;
//constexpr float INV_WIDTH = 1.f / WINDOW_WIDTH;
//constexpr float INV_HEIGHT = 1.f / WINDOW_HEIGHT;

// help functions
inline __device__ __host__ float fminf(float X, float Y, float Z) {
	return fminf(X, fminf(Y, Z));
}

inline __device__ __host__ float fmaxf(float X, float Y, float Z) {
	return fmaxf(X, fmaxf(Y, Z));
}

#ifndef hasItem(X, Y)
#define hasItem(X, Y) (X.find(Y) != X.end())
#endif

namespace glm {
template<int I, typename T, qualifier Q>
inline __device__ __host__ glm::vec<I, T, Q> min(const glm::vec<I, T, Q>& v1, const glm::vec<I, T, Q>& v2, const glm::vec<I, T, Q>& v3) {
	return glm::min(v1, glm::min(v2, v3));
}
template<int I, typename T, qualifier Q>
inline __device__ __host__ glm::vec<I, T, Q> max(const glm::vec<I, T, Q>& v1, const glm::vec<I, T, Q>& v2, const glm::vec<I, T, Q>& v3) {
	return glm::max(v1, glm::max(v2, v3));
}
}

inline __device__ __host__ glm::vec3 halfway(const glm::vec3& v, const glm::vec3& l) {
	return glm::normalize(v+l);
}

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

inline void hashCombine(std::size_t& seed) { }

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

// common data structures
namespace nagi {

struct Transform {
	glm::vec3 position{ 0.f, 0.f, 0.f };
	glm::vec3 rotation{ 0.f, 0.f, 0.f };
	glm::vec3 scale{ 1.f, 1.f, 1.f };
	glm::mat4 transformMat;
	glm::mat4 invTransformMat;
	glm::mat4 normalTransformMat;
	glm::mat4 invNormalTransformMat;
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
	float fov{ 1.f };
	float aspect{ 1.77777777777778f };
	float focusDistance{ 1.f };
	float apenture{ 0.f };
	glm::vec3 position{ 0.f, 0.f, 0.f };
	glm::vec3 upDir{ 0.f, -1.f, 0.f };
	glm::vec3 rightDir{ 1.f, 0.f, 0.f };
	glm::vec3 forwardDir{ 0.f, 0.f, 1.f };

	glm::vec3 filmOrigin;
	float halfW, halfH;
	float pixelWidth, pixelHeight, halfPixelWidth, halfPixelHeight;
};

struct BoundingBox { // AABB
	glm::vec3 min{ FLT_MAX, FLT_MAX, FLT_MAX };
	glm::vec3 max{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	glm::vec3 center;
	glm::vec3 halfExtent;
};
inline __device__ __host__ void updateBoundingBox(const glm::vec3& position, BoundingBox* box) {
	box->min = glm::min(position, box->min);
	box->max = glm::max(position, box->max);
	box->halfExtent = (box->max - box->min) * 0.5f;
	box->center = box->min + box->halfExtent;
}
inline __device__ __host__ void updateBoundingBox(const glm::vec3& min, const glm::vec3& max, BoundingBox* box) {
	box->min = min;
	box->max = max;
	box->halfExtent = (box->max - box->min) * 0.5f;
	box->center = box->min + box->halfExtent;
}

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 tangent;
	glm::vec2 uv;
	bool operator==(const Vertex& other) const { // don't compare tangent
		return position == other.position && normal == other.normal && uv == other.uv;
	}
	size_t hash() const {
		size_t seed = 0;
		//hashCombine(seed, position.x);
		//hashCombine(seed, position.y);
		//hashCombine(seed, position.z);
		//hashCombine(seed, normal.x);
		//hashCombine(seed, normal.y);
		//hashCombine(seed, normal.z);
		//hashCombine(seed, tangent.x);
		//hashCombine(seed, tangent.y);
		//hashCombine(seed, tangent.z);
		//hashCombine(seed, uv.x);
		//hashCombine(seed, uv.y);
		hashCombine(
			seed,
			position.x,
			position.y,
			position.z,
			normal.x,
			normal.y,
			normal.z,
			tangent.x,
			tangent.y,
			tangent.z,
			uv.x,
			uv.y);
		return seed;
	}
};

struct Triangle {
	int mtlIdx;
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
	int type{ PIXEL_TYPE_DIFFUSE };
	bool gbufferStored{ false };
	glm::vec3 color;
	int remainingBounces;
	int pixelIdx;
	int lastHit;
};

struct Object {
	//Transform transform;
	int trigIdxStart;
	int trigIdxEnd;
	int mtlIdx;
	BoundingBox bbox;
	int treeRoot;
};

struct Texture {
	int width;
	int height;
	int channels;
	cudaArray_t devArray;
	cudaTextureObject_t devTexture;
};

struct Material {
	int type;
	unsigned int textures{ 0 };
	//glm::vec3 emittance{ 0.f };
	glm::vec3 albedo{ 1.f }; // if light source, store emittance into albedo
	Texture baseTex;
	float roughness{ 1.f };
	Texture roughnessTex;
	float metallic{ 0.f };
	Texture metallicTex;
	Texture normalTex;
	Texture occlusionTex;
	float ior{ 1.5f };
};
inline __device__ __host__ bool hasTexture(const Material& mtl, unsigned int type) {
	return (mtl.textures & (1 << type)) != 0;
}
inline __device__ __host__ void addTextureType(Material& mtl, unsigned int type) {
	mtl.textures = (mtl.textures | (1 << type));
}

struct WindowSize {
	int width{ 1280 };
	int height{ 720 };
	int pixels{ 921600 };
	float invWidth{ 0.00078125 };
	float invHeight{ 0.00138888888888889f };
};

struct Configuration {
	int spp{ 64 };
	int maxBounce{ 8 };
	int denoiser{ 0 };
	float gamma{ 2.2f };
};

struct Scene {
	Configuration config;
	WindowSize window;
	Camera cam;

	BoundingBox bbox;
	Texture skybox;
	bool hasSkyBox{ false };

	std::vector<Object> objBuf;
	std::vector<Material> mtlBuf;
	std::vector<Triangle> trigBuf;
	unsigned int mtlTypes;
};
inline __device__ __host__ bool hasMaterial(const Scene& scene, unsigned int type) {
	return (scene.mtlTypes & (1 << type)) != 0;
}
inline __device__ __host__ void addMaterialType(Scene& scene, unsigned int type) {
	scene.mtlTypes = (scene.mtlTypes | (1 << type));
}

extern Scene scene; // global variable

}

namespace std {
template <>
struct hash<nagi::Vertex> {
	size_t operator()(nagi::Vertex const& vertex) const {
		return vertex.hash();
	}
};
}

#endif // !COMMON_CUH
