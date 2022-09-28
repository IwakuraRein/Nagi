#ifndef SCENE_LOADER_CUH
#define SCENE_LOADER_CUH

#include "common.cuh"

#include <json.hpp>

namespace nagi {

class SceneLoader {
public:
	SceneLoader(Scene& Scene, const std::string& FilePath) :scene{ Scene }, filePath{ FilePath } {}
	~SceneLoader() {}
	SceneLoader(const SceneLoader&) = delete;
	void operator=(const SceneLoader&) = delete;

	static nlohmann::json readJson(const std::string& filePath);

	void load();
	glm::ivec2 loadMesh(const std::string& meshPath, Object& obj);
	void loadConfig();
	void loadMaterials();
	void loadObjects();
	void loadCameras();

	bool printDetails{ false };
	Scene& scene;
	std::string filePath;
	nlohmann::json jFile;

	std::unordered_map<std::string, int> mtlIndices;
	//std::unordered_map<std::string, glm::ivec2> meshIndices;
};

}

#endif // !SCENE_LOADER_CUH