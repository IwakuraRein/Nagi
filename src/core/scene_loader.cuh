#ifndef SCENE_LOADER_CUH
#define SCENE_LOADER_CUH

#include "common.cuh"

#include <stb_image.h>
#include <json.hpp>

namespace nagi {

class SceneLoader {
public:
	SceneLoader(Scene& Scene, const std::string& FilePath) 
		:scene{ Scene }, filePath{ FilePath } {
		stbi_ldr_to_hdr_gamma(1.f); // disable gamma correction
	}
	~SceneLoader();
	SceneLoader(const SceneLoader&) = delete;
	void operator=(const SceneLoader&) = delete;

	static nlohmann::json readJson(const std::string& filePath);

	void load();
	glm::ivec2 loadMesh(const std::string& meshPath, Object& obj, const Transform& transform);
	static Texture loadTexture(const std::string& texPath, int desired_channel, bool srgb = false);
	void loadConfig();
	void loadMaterials();
	void loadObjects();
	void loadCameras();

	std::pair<glm::vec3, glm::vec3> calcTangent(Triangle& t); // calculate T and B
	std::unordered_map<int, std::pair<glm::vec3, glm::vec3>> tangents; // store faces' Ts and Bs
	std::unordered_map<Vertex, std::vector<int>> vert2face; // store unique vertices and their faces

	Scene& scene;
	std::string filePath;
	nlohmann::json jFile;

	std::unordered_map<std::string, int> mtlIndices;
	std::unordered_map<std::string, Texture> textures;
	//std::unordered_map<std::string, glm::ivec2> meshIndices;

	// avoid destroy one texture twice in deconstruction method
	std::vector<cudaArray_t> destroyedArrays;
	std::vector<cudaTextureObject_t> destroyedTextures;
};

}

#endif // !SCENE_LOADER_CUH