#include "scene_loader.cuh"
#include "io.cuh"

#include <tiny_obj_loader.h>

namespace nagi {

inline bool objComp(const Object& o1, const Object& o2) { return o1.mtlIdx < o2.mtlIdx; }

inline void updateTrigBoundingBox(Triangle& trig) {
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
		if (hasTexture(mtl, TEXTURE_TYPE_METALNESS)) {
			if (find(destroyedArrays.begin(), destroyedArrays.end(), mtl.metallicTex.devArray) == destroyedArrays.end()) {
				cudaFreeArray(mtl.metallicTex.devArray);
				destroyedArrays.push_back(mtl.metallicTex.devArray);
			}
			if (find(destroyedTextures.begin(), destroyedTextures.end(), mtl.metallicTex.devTexture) == destroyedTextures.end()) {
				cudaDestroyTextureObject(mtl.metallicTex.devTexture);
				destroyedTextures.push_back(mtl.metallicTex.devTexture);
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
			config.window.width = item.value()[0];
			config.window.height = item.value()[1];
			config.window.pixels = config.window.width * config.window.height;
			config.window.invWidth = 1.f / (float)config.window.width;
			config.window.invHeight = 1.f / (float)config.window.height;
		}
		if (item.key() == "alpha")
			config.alpha = item.value();
		if (item.key() == "gamma")
			config.gamma = item.value();
		if (item.key() == "sample rate")
			config.spp = item.value();
		if (item.key() == "max bounce")
			config.maxBounce = item.value();
		if (item.key() == "denoiser")
			config.denoiser = item.value();
	}
	scene.config = config;
	if (printDetails) std::cout << " done." << std::endl;
}

void SceneLoader::loadMaterials() {
	scene.mtlBuf.resize(jFile["materials"].size());
	int idx = 0;
	for (auto& material : jFile["materials"].items()) {
		if (printDetails) std::cout << "  Loading material " << material.key() << "...";
		auto& items = material.value();
		Material mtl{};
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

			if (hasItem(items, "emittance")) {
				auto& emittance = items["emittance"];
				mtl.albedo = glm::vec3{ emittance[0], emittance[1], emittance[2] };
			}

			else if (hasItem(items, "base texture")) {
				std::string texName(items["base texture"]);
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
				addTexture(mtl, TEXTURE_TYPE_BASE);
			}
			else {
				if (hasItem(items, "albedo")) {
					auto& albedo = items["albedo"];
					mtl.albedo = glm::vec3{ albedo[0], albedo[1], albedo[2] };
				}
			}

			if (hasItem(items, "roughness texture")) {
				std::string texName(items["base texture"]);
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
				addTexture(mtl, TEXTURE_TYPE_ROUGHNESS);
			}
			else {
				if (hasItem(items, "roughness")) {
					mtl.roughness = items["roughness"];
				}
			}

			if (hasItem(items, "metallic texture")) {
				std::string texName(items["base texture"]);
				texName = "roughness" + texName; // encoding texture's path with its type
				if (!hasItem(textures, texName)) {
					std::string texPath = items["metallic texture"];
					if (doesFileExist(texPath)); // do nothing
					else {
						std::string dir = strRightStrip(filePath, getFileName(filePath));
						if (doesFileExist(dir + texPath)) texPath = dir + texPath;
						else if (doesFileExist(dir + "textures/" + texPath)) texPath = dir + "textures/" + texPath;
						else throw std::runtime_error("Error: Metalness texture file doesn't exist.");
					}
					textures.emplace(texName, loadTexture(texPath, 1));
				}

				mtl.metallicTex = textures[texName];
				addTexture(mtl, TEXTURE_TYPE_METALNESS);
			}
			else {
				if (hasItem(items, "metallic")) {
					mtl.metallic = items["metallic"];
				}
			}

			//if (hasItem(items, "fresnel")) {
			//	auto& f = items["fresnel"];
			//	mtl.fresnel = glm::vec3{ f[0], f[1], f[2] };
			//}
			//if (hasItem(items, "ior")) {
			//	float ior = items["ior"];
			//	mtl.fresnel = glm::vec3{ (1.f - ior) / ((1.f + ior) * (1.f + ior)) };
			//}
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

	scene.bbox.min -= /*FLT_EPSILON*/0.01f;
	scene.bbox.max += /*FLT_EPSILON*/0.01f;

	// sort objects according to their materials
	std::sort(scene.objBuf.begin(), scene.objBuf.end(), objComp);
}

void SceneLoader::loadCameras() {
	if (printDetails) std::cout << "  Loading camera... ";
	Camera cam{};
	auto& camera = jFile["cameras"];
	if (hasItem(camera, "position")) {
		auto& position = camera["position"];
		cam.position = glm::vec3{ position[0], position[1], position[2] };
	}

	if (hasItem(camera, "fov")) {
		cam.fov = glm::radians((float)camera["fov"]);
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
	else cam.aspect = (float)scene.config.window.width * scene.config.window.invHeight;

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
		cam.forwardDir = glm::normalize(glm::vec3{ lookAt[0], lookAt[1], lookAt[2] } - cam.position);
	}
	cam.rightDir = glm::normalize(glm::cross(cam.forwardDir, cam.upDir));

	// multiplying a large number works around the insufficiency of float precision
	float halfh = tan(cam.fov / 2) * 10000.f;
	float halfw = halfh * cam.aspect;
	cam.screenOrigin = cam.forwardDir * 10000.f - cam.rightDir * halfw + cam.upDir * halfh;
	cam.pixelHeight = halfh * 2.f / scene.config.window.height;
	cam.pixelWidth = halfw * 2.f / scene.config.window.width;

	scene.cam = cam;

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
						attrib.texcoords[2 * vertInfo0.texcoord_index + 1]
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
						attrib.texcoords[2 * vertInfo1.texcoord_index + 1]
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
						attrib.texcoords[2 * vertInfo2.texcoord_index + 1]
				};
			}

			scene.trigBuf[trigIdx++] = std::move(trig);
		}
	}

	meshTrigIdx.y = scene.trigBuf.size() - 1;

	obj.bbox = BoundingBox{};
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

		updateBoundingBox(trig.vert0.position, &scene.bbox);
		updateBoundingBox(trig.vert1.position, &scene.bbox);
		updateBoundingBox(trig.vert2.position, &scene.bbox);
		updateBoundingBox(trig.vert0.position, &obj.bbox);
		updateBoundingBox(trig.vert1.position, &obj.bbox);
		updateBoundingBox(trig.vert2.position, &obj.bbox);
	}
	obj.bbox.min -= 0.01f;
	obj.bbox.max += 0.01f;
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