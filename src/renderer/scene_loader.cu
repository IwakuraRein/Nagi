#include "scene_loader.cuh"
#include "io.cuh"

#include <tiny_obj_loader.h>

namespace nagi {

bool objComp(const Object& o1, const Object& o2) { 
	return (o1.bbox.halfExtent.x * o1.bbox.halfExtent.y * o1.bbox.halfExtent.z) >
		(o2.bbox.halfExtent.x * o2.bbox.halfExtent.y * o2.bbox.halfExtent.z); }

void updateTrigBoundingBox(Triangle& trig) {
	// use epsilon to avoid bounding box having 0 volume.
	// FLT_EPSILON isn't enough. try using a larger number.
	updateBoundingBox(glm::min(trig.vert0.position, trig.vert1.position, trig.vert2.position) - /*FLT_EPSILON*/ 0.0001f,
		glm::max(trig.vert0.position, trig.vert1.position, trig.vert2.position) + /*FLT_EPSILON*/ 0.0001f,
		&trig.bbox);
}

nagi::SceneLoader::~SceneLoader() {
	// destroy all textures
	for (auto& mtl : scene.mtlBuf) {
		if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
			if (find(destroyedArrays.begin(), destroyedArrays.end(), mtl.baseTex.devArray) == destroyedArrays.end()) {
				cudaFreeArray(mtl.baseTex.devArray);
				destroyedArrays.push_back(mtl.baseTex.devArray);
			}
			if (find(destroyedTextures.begin(), destroyedTextures.end(), mtl.baseTex.devTexture) == destroyedTextures.end()) {
				cudaDestroyTextureObject(mtl.baseTex.devTexture);
				destroyedTextures.push_back(mtl.baseTex.devTexture);
			}
		}
		if (hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS)) {
			if (find(destroyedArrays.begin(), destroyedArrays.end(), mtl.roughnessTex.devArray) == destroyedArrays.end()) {
				cudaFreeArray(mtl.roughnessTex.devArray);
				destroyedArrays.push_back(mtl.roughnessTex.devArray);
			}
			if (find(destroyedTextures.begin(), destroyedTextures.end(), mtl.roughnessTex.devTexture) == destroyedTextures.end()) {
				cudaDestroyTextureObject(mtl.roughnessTex.devTexture);
				destroyedTextures.push_back(mtl.roughnessTex.devTexture);
			}
		}
		if (hasTexture(mtl, TEXTURE_TYPE_METALLIC)) {
			if (find(destroyedArrays.begin(), destroyedArrays.end(), mtl.metallicTex.devArray) == destroyedArrays.end()) {
				cudaFreeArray(mtl.metallicTex.devArray);
				destroyedArrays.push_back(mtl.metallicTex.devArray);
			}
			if (find(destroyedTextures.begin(), destroyedTextures.end(), mtl.metallicTex.devTexture) == destroyedTextures.end()) {
				cudaDestroyTextureObject(mtl.metallicTex.devTexture);
				destroyedTextures.push_back(mtl.metallicTex.devTexture);
			}
		}
		if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
			if (find(destroyedArrays.begin(), destroyedArrays.end(), mtl.normalTex.devArray) == destroyedArrays.end()) {
				cudaFreeArray(mtl.normalTex.devArray);
				destroyedArrays.push_back(mtl.normalTex.devArray);
			}
			if (find(destroyedTextures.begin(), destroyedTextures.end(), mtl.normalTex.devTexture) == destroyedTextures.end()) {
				cudaDestroyTextureObject(mtl.normalTex.devTexture);
				destroyedTextures.push_back(mtl.normalTex.devTexture);
			}
		}
	}
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
	std::cout << "Loading scene " << filePath << "...";
	if (!doesFileExist(filePath)) {
		std::string msg{ "Error: File " };
		msg += filePath;
		msg += " doesn't exist.";
		throw std::runtime_error(msg);
	}
	jFile = readJson(filePath);

	scene.bbox = BoundingBox{};

	loadConfig();
	loadCameras();
	loadMaterials();
	loadObjects();

	std::cout << " Done. Triangle count: " << scene.trigBuf.size() << std::endl;
}

void SceneLoader::loadConfig() {
	if (printDetails) std::cout << "  Loading configuration... ";
	Configuration config{};
	for (auto& item : jFile["graphics"].items()) {
		if (item.key() == "resolution") {
			scene.window.width = item.value()[0];
			scene.window.height = item.value()[1];
			scene.window.pixels = scene.window.width * scene.window.height;
			scene.window.invWidth = 1.f / (float)scene.window.width;
			scene.window.invHeight = 1.f / (float)scene.window.height;
		}
		if (item.key() == "gamma")
			config.gamma = item.value();
		if (item.key() == "sample rate")
			config.spp = item.value();
		if (item.key() == "max bounce")
			config.maxBounce = item.value();
		if (item.key() == "denoiser")
			config.denoiser = item.value();
		if (item.key() == "skybox") {
			std::string texName{ item.value() };
			texName = "skybox" + texName; // encoding texture's path with its type

			std::string texPath = item.value();
			if (!doesFileExist(texPath)) {
				std::string dir = strRightStrip(filePath, getFileName(filePath));
				if (doesFileExist(dir + texPath)) texPath = dir + texPath;
				else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
				else throw std::runtime_error("Error: Skybox image file doesn't exist.");
			}
			textures.emplace(texName, loadTexture(texPath, 4));
			scene.skybox = textures[texName];
			scene.hasSkyBox = true;
		}
	}
	scene.config = config;
	if (printDetails) std::cout << " done." << std::endl;
}

void SceneLoader::loadMaterials() {
	scene.mtlBuf.resize(jFile["materials"].size());
	scene.mtlTypes = 0;
	int idx = 0;
	for (auto& material : jFile["materials"].items()) {
		if (printDetails) std::cout << "  Loading material " << material.key() << "...";
		auto& items = material.value();
		Material mtl{};
		{
			if (hasItem(items, "type")) {
				std::string type = items["type"];
				if (type == "Lambert") {
					addMaterialType(scene, MTL_TYPE_LAMBERT);
					mtl.type = MTL_TYPE_LAMBERT;
				}
				else if (type == "Microfacet") {
					addMaterialType(scene, MTL_TYPE_MICROFACET);
					mtl.type = MTL_TYPE_MICROFACET;
				}
				else if (type == "Glass") {
					addMaterialType(scene, MTL_TYPE_GLASS);
					mtl.type = MTL_TYPE_GLASS;
				}
				else if (type == "Light Source") {
					addMaterialType(scene, MTL_TYPE_LIGHT_SOURCE);
					mtl.type = MTL_TYPE_LIGHT_SOURCE;
				}
				else if (type == "Mirror") {
					addMaterialType(scene, MTL_TYPE_MIRROR);
					mtl.type = MTL_TYPE_MIRROR;
				}
				else throw std::runtime_error("Error: Unknown material type.");
			}
			else throw std::runtime_error("Error: Material must specify its type.");

			if (hasItem(items, "emittance")) {
				auto& emittance = items["emittance"];
				mtl.albedo = glm::vec3{ emittance[0], emittance[1], emittance[2] };
			}

			else if (hasItem(items, "base texture")) {
				std::string texName{ items["base texture"] };
				texName = "base" + texName; // encoding texture's path with its type
				if (!hasItem(textures, texName)) {
					std::string texPath = items["base texture"];
					if (doesFileExist(texPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + texPath)) texPath = dir + texPath;
						else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
						else throw std::runtime_error("Error: Base texture file doesn't exist.");
					}
					if (!strEndWith(texName, ".hdr"))
						stbi_ldr_to_hdr_gamma(2.2f); // enable gamma correction for ldr picture
					textures.emplace(texName, loadTexture(texPath, 4));
					if (!strEndWith(texName, ".hdr"))
						stbi_ldr_to_hdr_gamma(1.f);
				}
				mtl.baseTex = textures[texName];
				addTextureType(mtl, TEXTURE_TYPE_BASE);
			}
			else {
				if (hasItem(items, "albedo")) {
					auto& albedo = items["albedo"];
					mtl.albedo = glm::vec3{ albedo[0], albedo[1], albedo[2] };
				}
			}

			if (hasItem(items, "roughness texture")) {
				auto hehe = items["roughness texture"];
				std::string texName{ items["roughness texture"] };
				texName = "roughness" + texName; // encoding texture's path with its type
				if (!hasItem(textures, texName)) {
					std::string texPath = items["roughness texture"];
					if (doesFileExist(texPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + texPath)) texPath = dir + texPath;
						else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
						else throw std::runtime_error("Error: Roughness texture file doesn't exist.");
					}
					textures.emplace(texName, loadTexture(texPath, 1));
				}

				mtl.roughnessTex = textures[texName];
				addTextureType(mtl, TEXTURE_TYPE_ROUGHNESS);
			}
			else {
				if (hasItem(items, "roughness")) {
					mtl.roughness = glm::clamp((float)items["roughness"], 0.05f, 1.f);
				}
			}

			if (hasItem(items, "metallic texture")) {
				std::string texName{ items["metallic texture"] };
				texName = "metallic" + texName; // encoding texture's path with its type
				if (!hasItem(textures, texName)) {
					std::string texPath = items["metallic texture"];
					if (doesFileExist(texPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + texPath)) texPath = dir + texPath;
						else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
						else throw std::runtime_error("Error: Metallic texture file doesn't exist.");
					}
					textures.emplace(texName, loadTexture(texPath, 1));
				}

				mtl.metallicTex = textures[texName];
				addTextureType(mtl, TEXTURE_TYPE_METALLIC);
			}
			else {
				if (hasItem(items, "metallic")) {
					mtl.metallic = items["metallic"];
				}
			}

			if (hasItem(items, "normal texture")) {
				std::string texName{ items["normal texture"] };
				texName = "normal" + texName; // encoding texture's path with its type
				if (!hasItem(textures, texName)) {
					std::string texPath = items["normal texture"];
					if (doesFileExist(texPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + texPath)) texPath = dir + texPath;
						else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
						else throw std::runtime_error("Error: Normal texture file doesn't exist.");
					}
					textures.emplace(texName, loadTexture(texPath, 4));
				}

				mtl.normalTex = textures[texName];
				addTextureType(mtl, TEXTURE_TYPE_NORMAL);
			}

			if (hasItem(items, "ior")) {
				mtl.ior = items["ior"];
			}
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
			Transform transform{};
			if (hasItem(items, "position")) {
				auto& position = items["position"];
				transform.position = glm::vec3{ position[0], position[1], position[2] };
			}

			if (hasItem(items, "rotation")) {
				auto& rotation = items["rotation"];
				transform.rotation = glm::radians(glm::vec3{ rotation[0], rotation[1], rotation[2] });
			}

			if (hasItem(items, "scale")) {
				auto& scale = items["scale"];
				transform.scale = glm::vec3{ scale[0], scale[1], scale[2] };
			}

			updateTransformMat(&transform);

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
					glm::ivec2 meshIdx = loadMesh(meshPath, obj, transform);
					//meshIndices.emplace(items["mesh"], meshIdx);
					obj.trigIdxStart = meshIdx.x;
					obj.trigIdxEnd = meshIdx.y;
				//}

			}
			else throw std::runtime_error("Error: Object must specify its mesh.");
		}
		scene.objBuf[idx++] = std::move(obj);
		if (printDetails) std::cout << " done." << std::endl;
	}

	scene.bbox.min -= /*FLT_EPSILON*/0.001f;
	scene.bbox.max += /*FLT_EPSILON*/0.001f;

	// sort objects according to their volume
	std::sort(scene.objBuf.begin(), scene.objBuf.end(), objComp);
}

void SceneLoader::loadCameras() {
	if (printDetails) std::cout << "  Loading camera... ";
	Camera cam{};
	auto& camera = jFile["camera"];
	if (hasItem(camera, "position")) {
		auto& position = camera["position"];
		cam.position = glm::vec3{ position[0], position[1], position[2] };
	}

	if (hasItem(camera, "fov")) {
		cam.fov = glm::radians((float)camera["fov"]);
	}

	if (hasItem(camera, "focus distance")) {
		cam.focusDistance = camera["focus distance"];
	}

	if (hasItem(camera, "near")) {
		cam.near = camera["near"];
	}

	if (hasItem(camera, "far")) {
		cam.far = camera["far"];
	}

	if (hasItem(camera, "aspect")) {
		if (camera["aspect"] <= 0.f) throw std::runtime_error("Error: Camera aspect must be postive.");
		cam.aspect = camera["aspect"];
	}
	else cam.aspect = (float)scene.window.width * scene.window.invHeight;

	if (hasItem(camera, "up")) {
		auto& up = camera["up"];
		cam.upDir = glm::normalize(glm::vec3{ up[0], up[1], up[2] });
	}
	if (hasItem(camera, "forward")) {
		auto& forward = camera["forward"];
		cam.forwardDir = glm::normalize(glm::vec3{ forward[0], forward[1], forward[2] });
	}
	else if (hasItem(camera, "look at")) {
		auto& lookAt = camera["look at"];
		glm::vec3 lookAtVec{ lookAt[0], lookAt[1], lookAt[2] };
		cam.focusDistance = glm::length(lookAtVec - cam.position);
		cam.forwardDir = glm::normalize(lookAtVec - cam.position);
	}
	cam.rightDir = glm::normalize(glm::cross(cam.forwardDir, cam.upDir));

	if (hasItem(camera, "f-number")) {
		cam.apenture = HALF_FILM_HEIGHT / glm::tan(cam.fov / 2.f) / (float)camera["f-number"] * 0.5f;
	}

	cam.halfH = cam.focusDistance * glm::tan(cam.fov/2.f) * CAMERA_MULTIPLIER;
	cam.halfW = cam.halfH * cam.aspect;
	cam.pixelHeight = cam.halfH * 2.f / scene.window.height;
	cam.pixelWidth = cam.halfW * 2.f / scene.window.width;
	cam.halfPixelHeight = cam.pixelHeight / 2.f;
	cam.halfPixelWidth = cam.pixelWidth / 2.f;
	cam.filmOrigin = cam.position + cam.forwardDir * cam.focusDistance * CAMERA_MULTIPLIER - cam.rightDir * cam.halfW + cam.upDir * cam.halfH;

	scene.cam = cam;

	if (printDetails) std::cout << " done." << std::endl;
}

std::pair<glm::vec3, glm::vec3> SceneLoader::calcTangent(Triangle& t) {
	const auto E1 = t.vert1.position - t.vert2.position;
	const auto E2 = t.vert2.position - t.vert0.position;
	const auto dUV1 = t.vert1.uv - t.vert0.uv;
	const auto dUV2 = t.vert2.uv - t.vert0.uv;

	float f = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);
	glm::vec3 T{
		f * (dUV2.y * E1.x - dUV1.y * E2.x),
		f * (dUV2.y * E1.y - dUV1.y * E2.y),
		f * (dUV2.y * E1.z - dUV1.y * E2.z),
	};
	T = glm::normalize(T);
	glm::vec3 B{
		f * (-dUV2.x * E1.x + dUV1.x * E2.x),
		f * (-dUV2.x * E1.y + dUV1.x * E2.y),
		f * (-dUV2.x * E1.z + dUV1.x * E2.z),
	};
	B = glm::normalize(B);
	return { T, B };
}

glm::ivec2 SceneLoader::loadMesh(const std::string& meshPath, Object& obj, const Transform& transform) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshPath.c_str())) {
		throw std::runtime_error(warn + err);
	}

	glm::ivec2 meshTrigIdx;
	meshTrigIdx.x = scene.trigBuf.size();

	// load mesh and calculate tangents
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
					vertInfo0.texcoord_index < 0 ||
					vertInfo1.vertex_index < 0 ||
					vertInfo1.normal_index < 0 ||
					vertInfo1.texcoord_index < 0 ||
					vertInfo2.vertex_index < 0 ||
					vertInfo2.normal_index < 0 ||
					vertInfo2.texcoord_index < 0)
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
				trig.vert0.uv = {
						attrib.texcoords[2 * vertInfo0.texcoord_index + 0],
						1.f - attrib.texcoords[2 * vertInfo0.texcoord_index + 1]
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
				trig.vert1.uv = {
						attrib.texcoords[2 * vertInfo1.texcoord_index + 0],
						1.f - attrib.texcoords[2 * vertInfo1.texcoord_index + 1]
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
				trig.vert2.uv = {
						attrib.texcoords[2 * vertInfo2.texcoord_index + 0],
						1.f - attrib.texcoords[2 * vertInfo2.texcoord_index + 1]
				};
			}

			scene.trigBuf[trigIdx] = std::move(trig);
			if (!hasItem(vert2face, trig.vert0)) {
				vert2face.emplace(trig.vert0, std::vector<int>{});
			}
			if (!hasItem(vert2face, trig.vert1)) {
				vert2face.emplace(trig.vert1, std::vector<int>{});
			}
			if (!hasItem(vert2face, trig.vert2)) {
				vert2face.emplace(trig.vert2, std::vector<int>{});
			}
			vert2face[trig.vert0].push_back(trigIdx);
			vert2face[trig.vert1].push_back(trigIdx);
			vert2face[trig.vert2].push_back(trigIdx);
			tangents.emplace(trigIdx, calcTangent(trig));
			trigIdx++;
		}
	}
	for (auto& pair : vert2face) {
		auto& vert = pair.first;
		glm::vec3 T{ 0.f };
		for (auto i : pair.second) {
			auto t = tangents[i].first;
			auto b = tangents[i].second;
			if (glm::dot(glm::cross(vert.normal, t), b) < 0.0f)
				t = t * -1.0f;
			T += t;
		}
		T = glm::normalize(T - vert.normal * glm::dot(vert.normal, T));
		for (auto i : pair.second) {
			auto& trig = scene.trigBuf[i];
			if (trig.vert0 == vert) trig.vert0.tangent = T;
			if (trig.vert1 == vert) trig.vert1.tangent = T;
			if (trig.vert2 == vert) trig.vert2.tangent = T;
		}
	}

	meshTrigIdx.y = scene.trigBuf.size() - 1;

	obj.bbox = BoundingBox{};
	// move everything to world space and update bounding boxes
	for (int i = meshTrigIdx.x; i <= meshTrigIdx.y; i++) {
		Triangle& trig = scene.trigBuf[i];
		vecTransform2(&trig.vert0.position, transform.transformMat);
		vecTransform2(&trig.vert1.position, transform.transformMat);
		vecTransform2(&trig.vert2.position, transform.transformMat);
		vecTransform2(&trig.vert0.normal, transform.normalTransformMat, 0.f);
		vecTransform2(&trig.vert1.normal, transform.normalTransformMat, 0.f);
		vecTransform2(&trig.vert2.normal, transform.normalTransformMat, 0.f);
		vecTransform2(&trig.vert0.tangent, transform.normalTransformMat, 0.f);
		vecTransform2(&trig.vert1.tangent, transform.normalTransformMat, 0.f);
		vecTransform2(&trig.vert2.tangent, transform.normalTransformMat, 0.f);
		trig.vert0.normal = glm::normalize(trig.vert0.normal);
		trig.vert1.normal = glm::normalize(trig.vert1.normal);
		trig.vert2.normal = glm::normalize(trig.vert2.normal);
		trig.vert0.tangent = glm::normalize(trig.vert0.tangent);
		trig.vert1.tangent = glm::normalize(trig.vert1.tangent);
		trig.vert2.tangent = glm::normalize(trig.vert2.tangent);

		updateTrigBoundingBox(trig);

		updateBoundingBox(trig.vert0.position, &scene.bbox);
		updateBoundingBox(trig.vert1.position, &scene.bbox);
		updateBoundingBox(trig.vert2.position, &scene.bbox);
		updateBoundingBox(trig.vert0.position, &obj.bbox);
		updateBoundingBox(trig.vert1.position, &obj.bbox);
		updateBoundingBox(trig.vert2.position, &obj.bbox);
	}
	obj.bbox.min -= 0.001f;
	obj.bbox.max += 0.001f;
	return meshTrigIdx;
}

// reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory
//            https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
Texture SceneLoader::loadTexture(const std::string& texPath, int ch, bool srgb) {
	if (!doesFileExist(texPath)) {
		throw std::runtime_error("Error: Image file doesn't exist.");
	}
	Texture tex;
	int chInFile;

	// use float buffer no matter it's ldr or hdr
	std::unique_ptr<float[]> buffer{ stbi_loadf(texPath.c_str(), &tex.width, &tex.height, &chInFile, ch) };
	if (buffer == nullptr) {
		throw std::runtime_error("Error: Can not load image file");
	}
	tex.channels = ch;
	//if (ch != chInFile) {
	//	std::cerr << "  Warning: Image file " << texPath << " has " << chInFile << " channels but the application requires a " << ch << "-channel image." << std::endl;
	//}
	cudaChannelFormatDesc channelDesc;
	
	switch (ch) {
	case 1:
		channelDesc = cudaCreateChannelDesc<float1>();
		break;
	case 4:
		channelDesc = cudaCreateChannelDesc<float4>();
		break;
	default:
		throw std::runtime_error("Error: Unsupported image channels.");
		break;
	}
	cudaMallocArray(&tex.devArray, &channelDesc, tex.width, tex.height);
	cudaMemcpyToArray(tex.devArray, 0, 0, buffer.get(), tex.width * tex.height * tex.channels * sizeof(float), cudaMemcpyHostToDevice);

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = tex.devArray;

	// Specify texture object parameters
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.sRGB = srgb ? 1 : -1;
	texDesc.normalizedCoords = 1;

	// Create texture object
	cudaCreateTextureObject(&tex.devTexture, &resDesc, &texDesc, NULL);

	return tex;
}

}