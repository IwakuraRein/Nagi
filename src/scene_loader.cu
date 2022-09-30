#include "scene_loader.cuh"

#include <tiny_obj_loader.h>

namespace nagi {

inline bool objComp(const Object& o1, const Object& o2) { return o1.mtlIdx < o2.mtlIdx; }

inline void updateTrigBoundingBox(Triangle& trig) {
	// use epsilon to avoid bounding box having 0 volume.
	// FLT_EPSILON isn't enough. try using a larger number.
	trig.bbox.min = glm::min(trig.vert0.position, glm::min(trig.vert1.position, trig.vert2.position)) - /*FLT_EPSILON*/ 0.0001f;
	trig.bbox.max = glm::max(trig.vert0.position, glm::max(trig.vert1.position, trig.vert2.position)) + /*FLT_EPSILON*/ 0.0001f;
}

inline void updateBoundingBox(const Vertex& vert, BoundingBox& bbox) {
	bbox.min = glm::min(vert.position, bbox.min);
	bbox.max = glm::max(vert.position, bbox.max);
}

nlohmann::json SceneLoader::readJson(const std::string& filePath) {
	std::ifstream file(filePath);
	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + filePath);
	}
	nlohmann::json j;
	file >> j;
	return j;
}

void SceneLoader::load() {
	std::cout << "Loading scene " << filePath << "..." << std::endl;
	if (!doesFileExist(filePath)) {
		std::string msg{ "Error: File " };
		msg += filePath;
		msg += " doesn't exist.";
		throw std::runtime_error(msg);
	}
	jFile = readJson(filePath);

	scene.bbox.min = glm::vec3{ FLT_MAX };
	scene.bbox.max = glm::vec3{ -FLT_MAX };

	loadConfig();
	loadCameras();
	loadMaterials();
	loadObjects();

	std::cout << "Loading scene finished. Triangle count: " << scene.trigBuf.size() << std::endl;
}

void SceneLoader::loadConfig() {
	if (printDetails) std::cout << "  Loading configuration... ";
	for (auto& item : jFile["graphics"].items()) {
		if (item.key() == "alpha")
			scene.config.alpha = item.value();
		if (item.key() == "gamma")
			scene.config.gamma = item.value();
		if (item.key() == "sample rate")
			scene.config.spp = item.value();
		if (item.key() == "max bounce")
			scene.config.maxBounce = item.value();
		if (item.key() == "denoiser")
			scene.config.denoiser = item.value();
	}
	if (printDetails) std::cout << " done." << std::endl;
}

void SceneLoader::loadMaterials() {
	scene.mtlBuf.resize(jFile["materials"].size());
	int idx = 0;
	for (auto& material : jFile["materials"].items()) {
		if (printDetails) std::cout << "  Loading material " << material.key() << "...";
		auto& items = material.value();
		Material mtl;
		{
			if (hasItem(items, "type")) {
				std::string type = items["type"];
				if (type == "Opaque") {
					mtl.type = MTL_TYPE_OPAQUE;
				}
				else if (type == "Transparent") {
					mtl.type = MTL_TYPE_TRANSPARENT;
				}
				else if (type == "Light Source") {
					mtl.type = MTL_TYPE_LIGHT_SOURCE;
				}
				else throw std::runtime_error("Error: Unknown material type.");
			}
			else throw std::runtime_error("Error: Material must specify its type.");
			if (hasItem(items, "albedo")) {
				auto& albedo = items["albedo"];
				mtl.albedo = glm::vec3{ albedo[0], albedo[1], albedo[2] };
			}
			else
				mtl.albedo = glm::vec3{ 1.f, 1.f, 1.f };
			if (hasItem(items, "roughness")) {
				mtl.roughness = items["roughness"];
			}
			else
				mtl.roughness = 1.f;
			if (hasItem(items, "metalness")) {
				mtl.metalness = items["metalness"];
			}
			else
				mtl.metalness = 0.f;
			if (hasItem(items, "transparent")) {
				mtl.transparent = items["transparent"];
			}
			else
				mtl.transparent = 0.f;
			if (hasItem(items, "ior")) {
				mtl.ior = items["ior"];
			}
			else
				mtl.ior = 1.f;
			if (hasItem(items, "emission")) {
				auto& emission = items["emission"];
				mtl.emission = glm::vec3{ emission[0], emission[1], emission[2] };
			}
			else
				mtl.emission = glm::vec3{ 0.f };
		}
		scene.mtlBuf[idx] = std::move(mtl);
		mtlIndices.emplace(material.key(), idx);
		idx++;
		if (printDetails) std::cout << " done." << std::endl;
	}
}

void SceneLoader::loadObjects() {
	scene.objBuf.resize(jFile["objects"].size());
	int idx = 0;
	for (auto& object : jFile["objects"].items()) {
		if (printDetails) std::cout << "  Loading object " << object.key() << "...";
		auto& items = object.value();
		Object obj;
		{
			if (hasItem(items, "position")) {
				auto& position = items["position"];
				obj.transform.position = glm::vec3{ position[0], position[1], position[2] };
			}
			else obj.transform.position = glm::vec3{ 0.f, 0.f, 0.f };

			if (hasItem(items, "rotation")) {
				auto& rotation = items["rotation"];
				obj.transform.rotation = glm::radians(glm::vec3{ rotation[0], rotation[1], rotation[2] });
			}
			else obj.transform.rotation = glm::vec3{ 0.f, 0.f, 0.f };

			if (hasItem(items, "scale")) {
				auto& scale = items["scale"];
				obj.transform.scale = glm::vec3{ scale[0], scale[1], scale[2] };
			}
			else obj.transform.scale = glm::vec3{ 1.f, 1.f, 1.f };

			updateTransformMat(&obj.transform);

			if (hasItem(items, "material")) {
				if (hasItem(mtlIndices, items["material"]))
					obj.mtlIdx = mtlIndices[items["material"]];
				else throw std::runtime_error("Error: Specified material doesn't exist.");
			}
			else throw std::runtime_error("Error: Object must specify its material.");

			if (hasItem(items, "mesh")) {
				std::string meshPath = items["mesh"];
				//if (hasItem(meshIndices, items["mesh"])) {
				//	// mesh file is already loaded
				//	obj.trigIdxStart = meshIndices[items["mesh"]].x;
				//	obj.trigIdxEnd = meshIndices[items["mesh"]].y;
				//}
				//else { // load new mesh file
					if (!strEndWith(meshPath, ".obj")) meshPath += ".obj";
					if (doesFileExist(meshPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + meshPath)) meshPath = dir + meshPath;
						else if (doesFileExist(dir + "models/" + meshPath)) meshPath = dir + "models/" + meshPath;
						else throw std::runtime_error("Error: Model file doesn't exist.");
					}
					glm::ivec2 meshIdx = loadMesh(meshPath, obj);
					//meshIndices.emplace(items["mesh"], meshIdx);
					obj.trigIdxStart = meshIdx.x;
					obj.trigIdxEnd = meshIdx.y;
					obj.bbox.min -= 0.0001f;
					obj.bbox.max += 0.0001f;
				//}

			}
			else throw std::runtime_error("Error: Object must specify its mesh.");
		}
		scene.objBuf[idx++] = std::move(obj);
		if (printDetails) std::cout << " done." << std::endl;
	}

	// sort objects according to their materials
	std::sort(scene.objBuf.begin(), scene.objBuf.end(), objComp);
}

void SceneLoader::loadCameras() {
	if (printDetails) std::cout << "  Loading camera... ";
	Camera& cam = scene.cam;
	auto& camera = jFile["cameras"];
	if (hasItem(camera, "position")) {
		auto& position = camera["position"];
		cam.position = glm::vec3{ position[0], position[1], position[2] };
	}
	else cam.position = glm::vec3{ 0.f, 0.f, 0.f };

	if (hasItem(camera, "fov")) {
		cam.fov = glm::radians((float)camera["fov"]);
	}
	else cam.fov = glm::radians(60.f);

	if (hasItem(camera, "near")) {
		cam.near = camera["near"];
	}
	else cam.near = 0.01f;

	if (hasItem(camera, "far")) {
		cam.far = camera["far"];
	}
	else cam.far = 1000.f;

	if (hasItem(camera, "aspect")) {
		if (camera["aspect"] <= 0.f) throw std::runtime_error("Error: Camera aspect must be postive.");
		cam.aspect = camera["aspect"];
	}
	else cam.aspect = WINDOW_WIDTH / WINDOW_HEIGHT;

	if (hasItem(camera, "up")) {
		auto& up = camera["up"];
		cam.upDir = glm::normalize(glm::vec3{ up[0], up[1], up[2] });
	}
	else cam.upDir = glm::vec3{ 0.f, -1.f, 0.f };
	if (hasItem(camera, "forward")) {
		auto& forward = camera["forward"];
		cam.forwardDir = glm::normalize(glm::vec3{ forward[0], forward[1], forward[2] });
	}
	else if (hasItem(camera, "look at")) {
		auto& lookAt = camera["look at"];
		cam.forwardDir = glm::normalize(glm::vec3{ lookAt[0], lookAt[1], lookAt[2] } - cam.position);
	}
	else cam.forwardDir = glm::vec3{ 0.f, 0.f, 1.f };
	cam.rightDir = glm::normalize(glm::cross(cam.forwardDir, cam.upDir));

	float halfh = tan(cam.fov / 2);
	float halfw = halfh * cam.aspect;
	cam.screenOrigin = cam.forwardDir - cam.rightDir * halfw + cam.upDir * halfh;
	cam.pixelHeight = halfh * 2.f * INV_HEIGHT;
	cam.pixelWidth = halfw * 2.f * INV_WIDTH;
	cam.halfPixelHeight = cam.pixelHeight / 2.f;
	cam.halfPixelWidth = cam.pixelWidth / 2.f;

	if (printDetails) std::cout << " done." << std::endl;
}

glm::ivec2 SceneLoader::loadMesh(const std::string& meshPath, Object& obj) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str())) {
		throw std::runtime_error(warn + err);
	}

	glm::ivec2 meshTrigIdx;
	meshTrigIdx.x = scene.trigBuf.size();

	// load mesh
	// shape.mesh -> face£¨triangle) -> vertex
	for (auto& shape : shapes) {
		// num_face_vertices.size(): how many faces does this mesh have
		// num_face_vertices[i]: is the face triangle, quad, or sth else
		int trigIdx = scene.trigBuf.size();
		scene.trigBuf.resize(scene.trigBuf.size() + shape.mesh.num_face_vertices.size());
		for (int i = 0; i < shape.mesh.num_face_vertices.size(); i++) {
			if (shape.mesh.num_face_vertices[i] != 3)
				throw std::runtime_error("Error: Only triangle face is supported.");
			Triangle trig;
			trig.mtlIdx = obj.mtlIdx;
			{ // load 3 vertices
				const auto& vertInfo0 = shape.mesh.indices[i * 3];
				const auto& vertInfo1 = shape.mesh.indices[i * 3 + 1];
				const auto& vertInfo2 = shape.mesh.indices[i * 3 + 2];
				if (vertInfo0.vertex_index < 0 ||
					vertInfo0.normal_index < 0 ||
					vertInfo1.vertex_index < 0 ||
					vertInfo1.normal_index < 0 ||
					vertInfo2.normal_index < 0 ||
					vertInfo2.normal_index < 0)
					throw std::runtime_error("Error: Inadequate vertex information.");
				trig.vert0.position = {
						attrib.vertices[3 * vertInfo0.vertex_index + 0],
						attrib.vertices[3 * vertInfo0.vertex_index + 1],
						attrib.vertices[3 * vertInfo0.vertex_index + 2],
				};
				trig.vert0.normal = {
						attrib.normals[3 * vertInfo0.normal_index + 0],
						attrib.normals[3 * vertInfo0.normal_index + 1],
						attrib.normals[3 * vertInfo0.normal_index + 2],
				};
				trig.vert1.position = {
						attrib.vertices[3 * vertInfo1.vertex_index + 0],
						attrib.vertices[3 * vertInfo1.vertex_index + 1],
						attrib.vertices[3 * vertInfo1.vertex_index + 2],
				};
				trig.vert1.normal = {
						attrib.normals[3 * vertInfo1.normal_index + 0],
						attrib.normals[3 * vertInfo1.normal_index + 1],
						attrib.normals[3 * vertInfo1.normal_index + 2],
				};
				trig.vert2.position = {
						attrib.vertices[3 * vertInfo2.vertex_index + 0],
						attrib.vertices[3 * vertInfo2.vertex_index + 1],
						attrib.vertices[3 * vertInfo2.vertex_index + 2],
				};
				trig.vert2.normal = {
						attrib.normals[3 * vertInfo2.normal_index + 0],
						attrib.normals[3 * vertInfo2.normal_index + 1],
						attrib.normals[3 * vertInfo2.normal_index + 2],
				};
			}

			scene.trigBuf[trigIdx++] = std::move(trig);
		}
	}

	meshTrigIdx.y = scene.trigBuf.size() - 1;

	obj.bbox.min = glm::vec3{ FLT_MAX };
	obj.bbox.max = glm::vec3{ -FLT_MAX };
	// move everything to world space and update bounding boxes
	for (int i = meshTrigIdx.x; i <= meshTrigIdx.y; i++) {
		Triangle& trig = scene.trigBuf[i];
		vecTransform2(&trig.vert0.position, obj.transform.transformMat);
		vecTransform2(&trig.vert1.position, obj.transform.transformMat);
		vecTransform2(&trig.vert2.position, obj.transform.transformMat);
		vecTransform2(&trig.vert0.normal, obj.transform.transformMat, 0.f);
		vecTransform2(&trig.vert1.normal, obj.transform.transformMat, 0.f);
		vecTransform2(&trig.vert2.normal, obj.transform.transformMat, 0.f);
		trig.vert0.normal = glm::normalize(trig.vert0.normal);
		trig.vert1.normal = glm::normalize(trig.vert1.normal);
		trig.vert2.normal = glm::normalize(trig.vert2.normal);

		updateTrigBoundingBox(trig);

		updateBoundingBox(trig.vert0, scene.bbox);
		updateBoundingBox(trig.vert1, scene.bbox);
		updateBoundingBox(trig.vert2, scene.bbox);
		updateBoundingBox(trig.vert0, obj.bbox);
		updateBoundingBox(trig.vert1, obj.bbox);
		updateBoundingBox(trig.vert2, obj.bbox);
	}

	return meshTrigIdx;
}

}