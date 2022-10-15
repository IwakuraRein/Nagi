#include "path_tracer.cuh"

#include <thrust/host_vector.h>  
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace nagi {

void PathTracer::initialize() {
	destroyBuffers();
	allocateBuffers();

	cudaMemcpy(devObjBuf, scene.objBuf.data(), scene.objBuf.size() * sizeof(Object), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devObjBuf failed.");
	cudaMemcpy(devMtlBuf, scene.mtlBuf.data(), scene.mtlBuf.size() * sizeof(Material), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devMtlBuf failed.");
	cudaMemcpy(devTrigBuf, scene.trigBuf.data(), scene.trigBuf.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devTrigBuf failed.");

	// no need to clear frame buffer. kernWriteFrameBuffer() will do this when spp=1
	//dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	//kernInitializeFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>>(window, devFrameBuf);
	//checkCUDAError("kernInitializeFrameBuffer failed.");

	cudaEventCreate(&timer_start);
	cudaEventCreate(&timer_end);
}

void PathTracer::allocateBuffers() {
	cudaMalloc((void**)&devObjBuf, scene.objBuf.size() * sizeof(Object));
	checkCUDAError("cudaMalloc devObjBuf failed.");
	cudaMalloc((void**)&devMtlBuf, scene.mtlBuf.size() * sizeof(Material));
	checkCUDAError("cudaMalloc devMtlBuf failed.");
	cudaMalloc((void**)&devTrigBuf, scene.trigBuf.size() * sizeof(Triangle));
	checkCUDAError("cudaMalloc devTrigBuf failed.");
	cudaMalloc((void**)&devFrameBuf, sizeof(float) * window.pixels * 3);
	checkCUDAError("cudaMalloc devFrameBuf failed.");
	cudaMalloc((void**)&devNormalBuf, sizeof(float) * window.pixels * 3);
	checkCUDAError("cudaMalloc devNormalBuf failed.");
	cudaMalloc((void**)&devAlbedoBuf, sizeof(float) * window.pixels * 3);
	checkCUDAError("cudaMalloc devAlbedoBuf failed.");
	cudaMalloc((void**)&devDepthBuf, sizeof(float) * window.pixels);
	checkCUDAError("cudaMalloc devDepthBuf failed.");
	cudaMalloc((void**)&devCurrentAlbedoBuf, sizeof(float) * window.pixels * 3);
	checkCUDAError("cudaMalloc devCurrentAlbedoBuf failed.");
	cudaMalloc((void**)&devCurrentDepthBuf, sizeof(float) * window.pixels);
	checkCUDAError("cudaMalloc devCurrentDepthBuf failed.");
	cudaMalloc((void**)&devVarianceBuf, sizeof(float) * window.pixels);
	checkCUDAError("cudaMalloc devVarianceBuf failed.");
	cudaMalloc((void**)&devRayPool1, window.pixels * sizeof(Path));
	checkCUDAError("cudaMalloc devRayPool1 failed.");
	cudaMalloc((void**)&devRayPool2, window.pixels * sizeof(Path));
	checkCUDAError("cudaMalloc devRayPool2 failed.");
	cudaMalloc((void**)&devTerminatedRays, window.pixels * sizeof(Path));
	checkCUDAError("cudaMalloc devTerminatedRays failed.");
	cudaMalloc((void**)&devResults1, window.pixels * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults1 failed.");
	cudaMalloc((void**)&devResults2, window.pixels * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults2 failed.");
}

void PathTracer::destroyBuffers() {
	if (devObjBuf) {
		cudaFree(devObjBuf);
		checkCUDAError("cudaFree devObjBuf failed.");
		devObjBuf = nullptr;
	}
	if (devMtlBuf) {
		cudaFree(devMtlBuf);
		checkCUDAError("cudaFree devMtlBuf failed.");
		devMtlBuf = nullptr;
	}
	if (devTrigBuf) {
		cudaFree(devTrigBuf);
		checkCUDAError("cudaFree devTrigBuf failed.");
		devTrigBuf = nullptr;
	}
	if (devRayPool1) {
		cudaFree(devRayPool1);
		checkCUDAError("cudaFree devRayPool1 failed.");
		devRayPool1 = nullptr;
	}
	if (devRayPool2) {
		cudaFree(devRayPool2);
		checkCUDAError("cudaFree devRayPool2 failed.");
		devRayPool2 = nullptr;
	}
	if (devTerminatedRays) {
		cudaFree(devTerminatedRays);
		checkCUDAError("cudaFree v failed.");
		devTerminatedRays = nullptr;
	}
	if (devResults1) {
		cudaFree(devResults1);
		checkCUDAError("cudaFree devResults1 failed.");
		devResults1 = nullptr;
	}
	if (devResults2) {
		cudaFree(devResults2);
		checkCUDAError("cudaFree devResults2 failed.");
		devResults2 = nullptr;
	}
	if (devFrameBuf) {
		cudaFree(devFrameBuf);
		checkCUDAError("cudaFree devFrameBuf failed.");
		devFrameBuf = nullptr;
	}
	if (devNormalBuf) {
		cudaFree(devNormalBuf);
		checkCUDAError("cudaFree devNormalBuf failed.");
		devNormalBuf = nullptr;
	}
	if (devAlbedoBuf) {
		cudaFree(devAlbedoBuf);
		checkCUDAError("cudaFree devAlbedoBuf failed.");
		devAlbedoBuf = nullptr;
	}
	if (devDepthBuf) {
		cudaFree(devDepthBuf);
		checkCUDAError("cudaFree devDepthBuf failed.");
		devDepthBuf = nullptr;
	}
	if (devCurrentAlbedoBuf) {
		cudaFree(devCurrentAlbedoBuf);
		checkCUDAError("cudaFree devCurrentAlbedoBuf failed.");
		devCurrentAlbedoBuf = nullptr;
	}
	if (devCurrentDepthBuf) {
		cudaFree(devCurrentDepthBuf);
		checkCUDAError("cudaFree devCurrentDepthBuf failed.");
		devCurrentDepthBuf = nullptr;
	}
	if (devVarianceBuf) {
		cudaFree(devVarianceBuf);
		checkCUDAError("cudaFree devVarianceBuf failed.");
		devVarianceBuf = nullptr;
	}
}

PathTracer::~PathTracer() {
	destroyBuffers();
	cudaEventDestroy(timer_start);
	cudaEventDestroy(timer_end);
}

// intersection test -> compact rays -> sort rays according to material -> compute color -> compact rays -> intersection test...
void PathTracer::iterate() {
	std::chrono::steady_clock::time_point timer;
	std::cout << "  Begin iteration " << spp << ". " << scene.config.spp - spp << " remaining." << std::endl;
	timer = std::chrono::high_resolution_clock::now();
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeRays<<<blocksPerGrid, BLOCK_SIZE>>> (window, spp, devRayPool1, scene.config.maxBounce, scene.cam, spp != 1);
	checkCUDAError("kernInitializeRays failed.");
	int bounce = 0;
	int remainingRays = window.pixels;
	while (true) {
		remainingRays = intersectionTest(remainingRays);
		if (bounce == 0) std::cout << "    First intersection test: " << intersectionTime << " ms." << std::endl;
		bounce++;

		//std::cout << remainingRays << " ";
		if (remainingRays <= 0) break;

		//sortRays(remainingRays);

		tik();
		if (bounce == 1 && scene.hasSkyBox) {
			generateSkyboxAlbedo(window.pixels - remainingRays, spp);
		}
		if (bounce <= MAX_GBUFFER_BOUNCE) {
			generateGbuffer(remainingRays, spp, bounce);
		}
		gbufferTime += tok();

		remainingRays = shade(remainingRays, spp);
		//std::cout << remainingRays << std::endl;
		if (remainingRays <= 0) break;
	}
	if (scene.hasSkyBox) {
		shadeWithSkybox();
	}
	writeFrameBuffer(spp);
	terminatedRayNum = 0;
	spp++;

	if (printDetails) {
		float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
		std::cout << "    Intersection test: " << intersectionTime / 1000.f << " seconds." << std::endl;
		std::cout << "    Shading: " << shadingTime / 1000.f << " seconds." << std::endl; 
		std::cout << "    Compaction : " << compactionTime / 1000.f << " seconds." << std::endl;
		std::cout << "    Gbuffer generation : " << gbufferTime / 1000.f << " seconds." << std::endl;

		std::cout << "  Iteration " << spp - 1 << " finished. Time cost: " << runningTime <<
			" seconds. Time Remaining: " << runningTime * (scene.config.spp - spp + 1) << " seconds." << std::endl; 

		intersectionTime = 0.f;
		compactionTime = 0.f;
		shadingTime = 0.f;
		gbufferTime = 0.f;
	}
}

__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec2 pickedBaryCentric, baryCentric;
	Triangle pickedTrig;
	int pickedMtlIdx{ -1 };
	float dist;
	float minDist{ FLT_MAX };
	for (int i = trigIdxStart; i <= trigIdxEnd; i++) {
		Triangle trig = trigBuf[i];
		if (rayBoxIntersect(r, trig.bbox, dist)) {
			if (dist < minDist) {
				if (rayTrigIntersect(r, trig, dist, baryCentric)) {
					//if (glm::intersectRayTriangle(r.origin, r.dir, trig.vert0.position, trig.vert1.position, trig.vert2.position, baryCentric, dist)) {
					if (dist > 0.f && dist < minDist) {
						minDist = dist;
						pickedTrig = trig;
						pickedBaryCentric = baryCentric;
						pickedMtlIdx = trig.mtlIdx;
					}
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.normal + pickedBaryCentric.x * pickedTrig.vert1.normal + pickedBaryCentric.y * pickedTrig.vert2.normal;
	result.tangent = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.tangent + pickedBaryCentric.x * pickedTrig.vert1.tangent + pickedBaryCentric.y * pickedTrig.vert2.tangent;
	result.uv = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.uv + pickedBaryCentric.x * pickedTrig.vert1.uv + pickedBaryCentric.y * pickedTrig.vert2.uv;
	result.position = r.origin + r.dir * (minDist - REFLECT_OFFSET);
	//r.origin = r.origin + r.dir * (minDist - REFLECT_OFFSET);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;
}

__global__ void kernObjIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec3 normal, position, tangent;
	glm::vec2 pickedBaryCentric, baryCentric;
	Triangle pickedTrig;
	int pickedMtlIdx{ -1 };
	float dist;
	float minDist{ FLT_MAX };
	for (int i = 0; i < objNum; i++) {
		Object obj = objBuf[i];
		if (rayBoxIntersect(r, obj.bbox, dist)) {
			if (dist < minDist) {
				for (int j = obj.trigIdxStart; j <= obj.trigIdxEnd; j++) {
					Triangle trig = trigBuf[j];
					if (rayBoxIntersect(r, trig.bbox, dist)) {
						if (rayTrigIntersect(r, trig, dist, baryCentric)) {
							if (dist > 0.f && dist < minDist) {
								minDist = dist;
								pickedBaryCentric = baryCentric;
								pickedTrig = trig;
								pickedMtlIdx = trig.mtlIdx;
							}
						}
					}
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.normal + pickedBaryCentric.x * pickedTrig.vert1.normal + pickedBaryCentric.y * pickedTrig.vert2.normal;
	result.tangent = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.tangent + pickedBaryCentric.x * pickedTrig.vert1.tangent + pickedBaryCentric.y * pickedTrig.vert2.tangent;
	result.uv = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.uv + pickedBaryCentric.x * pickedTrig.vert1.uv + pickedBaryCentric.y * pickedTrig.vert2.uv;
	result.position = r.origin + r.dir * (minDist - REFLECT_OFFSET);
	//r.origin = r.origin + r.dir * (minDist - REFLECT_OFFSET);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;

}

__global__ void kernBVHIntersectTest(
	int rayNum, Path* rayPool, BVH::Node root, BVH::Node* treeBuf, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec2 pickedBaryCentric, baryCentric;
	Triangle pickedTrig;
	int pickedMtlIdx{ -1 };
	float trigDist;
	float minTrigD{ FLT_MAX };
	float boxD;
	//Bound lastTrigBox{};

	BVH::Node stack[MAX_TREE_DEPTH + 1];
	int searched[MAX_TREE_DEPTH + 1] = { 0 };
	int ptr = 0;
	stack[0] = root;

	glm::vec3 childMin, childMax;
	int child;
	while (ptr >= 0) {
		BVH::Node& node = stack[ptr];
		if (!node.leaf) {
			if (searched[ptr] < 2) {
				if (searched[ptr] == 0) {
					childMin = node.leftMin;
					childMax = node.leftMax;
					child = node.left;
				}
				if (searched[ptr] == 1) {
					childMin = node.rightMin;
					childMax = node.rightMax;
					child = node.right;
				}
				searched[ptr]++;
				if (rayBoxIntersect(r, childMin, childMax, boxD)) {
					if (boxD < minTrigD) {
						ptr++;
						stack[ptr] = treeBuf[child];
						searched[ptr] = 0;
						continue;
					}
				}
			}
			else ptr--;
		}
		else {
			for (int i = node.left; i <= node.right; i++) {
				Triangle trig = trigBuf[i];
				if (rayBoxIntersect(r, trig.bbox, trigDist)) {
					//if (trigDist >= minTrigD) {
					//	if (!boxBoxIntersect(trig.bbox, lastTrigBox)) continue;
					//}
					if (rayTrigIntersect(r, trig, trigDist, baryCentric)) {
					//if (glm::intersectRayTriangle(r.origin, r.dir, trig.vert0.position, trig.vert1.position, trig.vert2.position, baryCentric, trigDist)) {
						if (trigDist > 0.f && trigDist < minTrigD) {
							minTrigD = trigDist;
							pickedBaryCentric = baryCentric;
							pickedTrig = trig;
							pickedMtlIdx = trig.mtlIdx;
							//lastTrigBox = trig.bbox;
						}
					}
				}
			}
			ptr--;
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.normal + pickedBaryCentric.x * pickedTrig.vert1.normal + pickedBaryCentric.y * pickedTrig.vert2.normal;
	result.tangent = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.tangent + pickedBaryCentric.x * pickedTrig.vert1.tangent + pickedBaryCentric.y * pickedTrig.vert2.tangent;
	result.uv = (1 - pickedBaryCentric.x - pickedBaryCentric.y) * pickedTrig.vert0.uv + pickedBaryCentric.x * pickedTrig.vert1.uv + pickedBaryCentric.y * pickedTrig.vert2.uv;
	result.position = r.origin + r.dir * (minTrigD - REFLECT_OFFSET);
	//r.origin = r.origin + r.dir * (minTrigD - REFLECT_OFFSET);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;
}

int PathTracer::intersectionTest(int rayNum) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);

	tik();
	//kernTrigIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, 0, scene.trigBuf.size()-1, devTrigBuf, devResults1);
	//kernObjIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, scene.objBuf.size(), devObjBuf, devTrigBuf, devResults1);
	kernBVHIntersectTest<<<blocksPerGrid, BLOCK_SIZE>>> (
		rayNum, devRayPool1, bvh.tree[bvh.treeRoot], bvh.devTree, devTrigBuf, devResults1);
	intersectionTime += tok();
	checkCUDAError("kernBVHIntersectTest failed.");

	tik();
	rayNum = compactRays(rayNum, devRayPool1, devRayPool2, devResults1, devResults2);
	compactionTime += tok();

	std::swap(devRayPool1, devRayPool2);
	std::swap(devResults1, devResults2);

	return rayNum;
}

// sort rays according to materials
void PathTracer::sortRays(int rayNum) {
	thrust::device_ptr<Path> tRays{ devRayPool1 };
	thrust::device_ptr<IntersectInfo> tResults{ devResults1 };

	thrust::stable_sort_by_key(tResults, tResults + rayNum, tRays, IntersectionComp());
	checkCUDAError("thrust::stable_sort_by_key failed.");
	thrust::stable_sort(tResults, tResults + rayNum, IntersectionComp());
	checkCUDAError("thrust::stable_sort failed.");
}

__global__ void kernGenerateGbuffer(
	int rayNum, float currentSpp, int bounce, glm::vec3 camPos, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf, 
	float* currentAlbedoBuf, float* currentDepthBuf, float* albedoBuf, float* normalBuf, float* depthBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Path p = rayPool[idx];
	if (p.gbufferStored) return;
	int pixel = p.pixelIdx;
	IntersectInfo intersect = intersections[idx];
	Material mtl = mtlBuf[intersect.mtlIdx];

	glm::vec3 normal;
	glm::vec3 albedo;
	float depth = glm::length(intersect.position - p.ray.origin);
	if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
		glm::mat3 TBN = glm::mat3(intersect.tangent, glm::cross(intersect.normal, intersect.tangent), intersect.normal);
		float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersect.uv.x, intersect.uv.y);
		glm::vec3 bump{ texVal.x * 2.f - 1.f, texVal.y * 2.f - 1.f, 1.f };
		bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
		normal = glm::normalize(TBN * bump);
	}
	else normal = intersect.normal;

	if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
		float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersect.uv.x, intersect.uv.y);
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
	}
	else albedo = mtl.albedo;

	if (bounce == 1) {
		if (mtl.type == MTL_TYPE_SPECULAR) {
			float metallic;
			if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
				metallic = tex2D<float>(mtl.metallicTex.devTexture, intersect.uv.x, intersect.uv.y);
			}
			else metallic = mtl.metallic;
			if (metallic >= 0.5f)
				p.type = PIXEL_TYPE_SPECULAR;
			else
				p.type = PIXEL_TYPE_GLOSSY;
		}
		if (mtl.type == MTL_TYPE_MICROFACET) {
			if (!hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS) && mtl.roughness <= 0.1f) {
				p.type = PIXEL_TYPE_GLOSSY;
			}
			else if (hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS)) {
				if (tex2D<float>(mtl.roughnessTex.devTexture, intersect.uv.x, intersect.uv.y) <= 0.1f) {
					p.type = PIXEL_TYPE_GLOSSY;
				}
			}
		}
		if (mtl.type == MTL_TYPE_GLASS) p.type = PIXEL_TYPE_GLOSSY;

		if (p.type == PIXEL_TYPE_SPECULAR) {
			currentAlbedoBuf[pixel * 3] = albedo.x;
			currentAlbedoBuf[pixel * 3 + 1] = albedo.y;
			currentAlbedoBuf[pixel * 3 + 2] = albedo.z;
			if (currentSpp == 1.f)currentDepthBuf[pixel] = depth;
		}
		else if (p.type == PIXEL_TYPE_GLOSSY) {
			currentAlbedoBuf[pixel * 3] = albedo.x;
			currentAlbedoBuf[pixel * 3 + 1] = albedo.y;
			currentAlbedoBuf[pixel * 3 + 2] = albedo.z;
			//currentDepthBuf[pixel] = depth;

			if (currentSpp == 1.f) {
				depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + depth) / currentSpp;
				normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
				normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
				normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
			}
		}
		else {
			p.type == PIXEL_TYPE_DIFFUSE;
			p.gbufferStored = true;
			albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + albedo.x) / currentSpp;
			albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + albedo.y) / currentSpp;
			albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + albedo.z) / currentSpp;

			if (currentSpp == 1.f) {
				depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + depth) / currentSpp;

				normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
				normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
				normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
			}
		}

		rayPool[idx] = p;
	}

	else {
		if (p.type == PIXEL_TYPE_GLOSSY) {
			bool hitDiffuse{ false };

			if (mtl.type == MTL_TYPE_LAMBERT || mtl.type == MTL_TYPE_LIGHT_SOURCE) {
				hitDiffuse = true;
				rayPool[idx] = p;
			}
			if (mtl.type == MTL_TYPE_MICROFACET) {
				if (!hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS) && mtl.roughness > 0.1f) {
					hitDiffuse = true;
					rayPool[idx] = p;
				}
				else if (hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS)) {
					if (tex2D<float>(mtl.roughnessTex.devTexture, intersect.uv.x, intersect.uv.y) > 0.1f) {
						hitDiffuse = true;
						rayPool[idx] = p;
					}
				}
			}

			if (hitDiffuse || bounce == MAX_GBUFFER_BOUNCE) {
				p.gbufferStored = true;
				rayPool[idx] = p;
				albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3] * albedo.x) / currentSpp;
				albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 1] * albedo.y) / currentSpp;
				albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 2] * albedo.z) / currentSpp;
			}
			else {
				currentAlbedoBuf[pixel * 3] *= albedo.x;
				currentAlbedoBuf[pixel * 3 + 1] *= albedo.y;
				currentAlbedoBuf[pixel * 3 + 2] *= albedo.z;
			}
		}
		if (p.type == PIXEL_TYPE_SPECULAR) {
			if (mtl.type != MTL_TYPE_SPECULAR || bounce == MAX_GBUFFER_BOUNCE) {
				p.gbufferStored = true;
				rayPool[idx] = p;
				albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3] * albedo.x) / currentSpp;
				albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 1] * albedo.y) / currentSpp;
				albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 2] * albedo.z) / currentSpp;
				if (currentSpp == 1.f) {
					depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + currentDepthBuf[pixel] + depth) / currentSpp;
					normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
					normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
					normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
				}
			}
			else {
				currentAlbedoBuf[pixel * 3] *= albedo.x;
				currentAlbedoBuf[pixel * 3 + 1] *= albedo.y;
				currentAlbedoBuf[pixel * 3 + 2] *= albedo.z;
				if (currentSpp == 1.f) currentDepthBuf[pixel] += depth;
			}
		}
	}
}
void PathTracer::generateGbuffer(int rayNum, int spp, int bounce) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
	//if (spp == 1) {
	//	kernGenerateNormalDepth(
	//		rayNum, bounce, scene.cam.position, devRayPool1, devResults1, devMtlBuf, devNormalBuf, devDepthBuf);
	//}
	//kernGenerateAlbedo(rayNum, float(spp), bounce, devRayPool1, devResults1, devMtlBuf, devCurrentAlbedoBuf, devAlbedoBuf);
	kernGenerateGbuffer<<<blocksPerGrid, BLOCK_SIZE>>>(
		rayNum, (float)spp, bounce, scene.cam.position, devRayPool1, devResults1, devMtlBuf, devCurrentAlbedoBuf, devCurrentDepthBuf, devAlbedoBuf, devNormalBuf, devDepthBuf);
}


// compute color and generate new ray direction
__global__ void kernShadeLambert(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_LAMBERT) return;

	Path p = rayPool[idx];
	p.ray.origin = intersection.position;

	if (p.remainingBounces == 0) {
		p.color = glm::vec3{ 0.f };
	}
	else if (glm::dot(intersection.normal, p.ray.dir) >= 0.f) {
		p.color = glm::vec3{ 0.f };
		p.remainingBounces = 0;
	}
	else {
		auto rng = makeSeededRandomEngine(spp, idx, 0);
		thrust::uniform_real_distribution<double> u01(0.f, 1.f);
		glm::vec3 normal, albedo;
		glm::vec3 wo, bsdf;
		float pdf;
		if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
			glm::mat3 TBN = glm::mat3(intersection.tangent, glm::cross(intersection.normal, intersection.tangent), intersection.normal);
			float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersection.uv.x, intersection.uv.y);
			glm::vec3 bump{ -texVal.x * 2.f + 1.f, -texVal.y * 2.f + 1.f, texVal.z * 2.f - 1.f };
			bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
			normal = glm::normalize(TBN * bump);
		}
		else normal = intersection.normal;
		if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
			float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersection.uv.x, intersection.uv.y);
			albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
		}
		else albedo = mtl.albedo;
		wo = cosHemisphereSampler(normal, pdf, u01(rng), u01(rng));
		if (pdf < PDF_EPSILON) {
			//p.lastHit = -1;
			p.color = glm::vec3{ 0.f };
			p.remainingBounces = 0;
		}
		else {
			bsdf = lambertBrdf(p.ray.dir, wo, normal, albedo);
			p.color = p.color * bsdf / pdf; // lambert is timed inside the bsdf
			p.ray.dir = wo;
			p.ray.invDir = 1.f / wo;
		}
		p.remainingBounces--;
	}
	rayPool[idx] = p;
}
__global__ void kernShadeSpecular(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_SPECULAR) return;

	Path p = rayPool[idx];
	p.ray.origin = intersection.position;

	if (p.remainingBounces == 0 || glm::dot(intersection.normal, p.ray.dir) >= 0.f) {
		p.color = glm::vec3{ 0.f };
		p.remainingBounces = 0;
	}
	else {
		auto rng = makeSeededRandomEngine(spp, idx, 0);
		thrust::uniform_real_distribution<double> u01(0.f, 1.f);
		glm::vec3 normal, albedo;
		float metallic;
		if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
			glm::mat3 TBN = glm::mat3(intersection.tangent, glm::cross(intersection.normal, intersection.tangent), intersection.normal);
			float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersection.uv.x, intersection.uv.y);
			glm::vec3 bump{ -texVal.x * 2.f + 1.f, -texVal.y * 2.f + 1.f, texVal.z * 2.f - 1.f };
			bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
			normal = glm::normalize(TBN * bump);
		}
		else normal = intersection.normal;
		if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
			float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersection.uv.x, intersection.uv.y);
			albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
		}
		else albedo = mtl.albedo;
		if (hasTexture(mtl, TEXTURE_TYPE_METALLIC)) {
			metallic = tex2D<float>(mtl.metallicTex.devTexture, intersection.uv.x, intersection.uv.y);
		}
		else metallic = mtl.metallic;

		float pdf;
		bool specular;
		glm::vec3 wo = reflectSampler(metallic, albedo, p.ray.dir, normal, u01(rng), u01(rng), u01(rng), pdf, specular);
		if (pdf < PDF_EPSILON) {
			p.color = glm::vec3{ 0.f };
			p.remainingBounces = 0;
		}
		else {
			if (!specular)
				albedo = lambertBrdf(p.ray.dir, wo, normal, albedo); // lambert is timed inside the bsdf
			p.color = p.color * albedo / pdf;
			p.ray.dir = wo;
			p.ray.invDir = 1.f / p.ray.dir;

			p.remainingBounces--;
		}
	}
	rayPool[idx] = p;
}
__global__ void kernShadeLightSource(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_LIGHT_SOURCE) return;

	Path p = rayPool[idx];
	p.ray.origin = intersection.position;

	p.remainingBounces = 0;
	if (glm::dot(intersection.normal, p.ray.dir) >= 0.f) {
		p.color = glm::vec3{ 0.f };
	}
	else {
		p.color *= mtl.albedo;
	}
	rayPool[idx] = p;
}
__global__ void kernShadeGlass(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_GLASS) return;

	Path p = rayPool[idx];
	p.ray.origin = intersection.position;

	if (p.remainingBounces == 0) {
		p.color = glm::vec3{ 0.f };
	}
	else {
		auto rng = makeSeededRandomEngine(spp, idx, 0);
		thrust::uniform_real_distribution<double> u01(0.f, 1.f);
		glm::vec3 normal, albedo;
		glm::vec3 wo;
		if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
			glm::mat3 TBN = glm::mat3(intersection.tangent, glm::cross(intersection.normal, intersection.tangent), intersection.normal);
			float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersection.uv.x, intersection.uv.y);
			glm::vec3 bump{ -texVal.x * 2.f + 1.f, -texVal.y * 2.f + 1.f, texVal.z * 2.f - 1.f };
			bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
			normal = glm::normalize(TBN * bump);
		}
		else normal = intersection.normal;
		if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
			float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersection.uv.x, intersection.uv.y);
			albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
		}
		else albedo = mtl.albedo;

		wo = refractSampler(mtl.ior, p.ray.dir, normal, u01(rng));
		p.color = p.color * albedo;
		p.ray.origin += wo * REFRACT_OFFSET;
		p.ray.dir = wo;
		p.ray.invDir = 1.f / wo;
		p.remainingBounces--;
	}
	rayPool[idx] = p;
}
__global__ void kernShadeMicrofacet(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_MICROFACET) return;

	Path p = rayPool[idx];
	p.ray.origin = intersection.position;


	if (p.remainingBounces == 0) {
		p.color = glm::vec3{ 0.f };
	}
	else if (glm::dot(intersection.normal, p.ray.dir) >= 0.f) {
		p.color = glm::vec3{ 0.f };
		p.remainingBounces = 0;
	}
	else {
		auto rng = makeSeededRandomEngine(spp, idx, 0);
		thrust::uniform_real_distribution<double> u01(0.f, 1.f);
		float samplingRnd1 = u01(rng);
		float samplingRnd2 = u01(rng);
		glm::vec3 normal, albedo;
		glm::vec3 wo, bsdf;
		float pdf;
		if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
			glm::mat3 TBN = glm::mat3(intersection.tangent, glm::cross(intersection.normal, intersection.tangent), intersection.normal);
			float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersection.uv.x, intersection.uv.y);
			glm::vec3 bump{ -texVal.x * 2.f + 1.f, -texVal.y * 2.f + 1.f, texVal.z * 2.f - 1.f };
			bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
			normal = glm::normalize(TBN * bump);
		}
		else normal = intersection.normal;
		if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
			float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersection.uv.x, intersection.uv.y);
			albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
		}
		else albedo = mtl.albedo;
		float metallic, roughness;
		if (hasTexture(mtl, TEXTURE_TYPE_METALLIC)) {
			metallic = tex2D<float>(mtl.metallicTex.devTexture, intersection.uv.x, intersection.uv.y);
		}
		else metallic = mtl.metallic;
		if (hasTexture(mtl, TEXTURE_TYPE_ROUGHNESS)) {
			roughness = tex2D<float>(mtl.roughnessTex.devTexture, intersection.uv.x, intersection.uv.y);
		}
		else roughness = mtl.roughness;

		float r = u01(rng);
		float s = 0.5f + metallic / 2.f;
		float pdf2;
		wo = ggxImportanceSampler(roughness, p.ray.dir, normal, pdf, samplingRnd1, samplingRnd2);
		glm::vec3 wo2 = cosHemisphereSampler(normal, pdf2, samplingRnd1, samplingRnd2);
		if (r > s) wo = wo2;
		pdf = s * pdf + (1.f - s) * pdf2;

		if (glm::dot(wo, normal) <= FLT_EPSILON || pdf < PDF_EPSILON) {
			p.color = glm::vec3{ 0.f };
			p.remainingBounces = 0;
		}
		else {
			bsdf = microfacetBrdf(p.ray.dir, wo, normal, albedo, metallic, roughness);
			p.color = p.color * bsdf / pdf; // lambert is timed inside the bsdf
			p.ray.dir = wo;
			p.ray.invDir = 1.f / wo;
		}

		p.remainingBounces--;
	}
	rayPool[idx] = p;
}

int PathTracer::shade(int rayNum, int spp) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);

	tik();
	if (hasMaterial(scene, MTL_TYPE_LIGHT_SOURCE))
		kernShadeLightSource<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_LAMBERT))
		kernShadeLambert<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_SPECULAR))
		kernShadeSpecular<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_GLASS))
		kernShadeGlass<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_MICROFACET))
		kernShadeMicrofacet<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);
	shadingTime += tok();

	checkCUDAError("kernShading failed.");

	tik();
	rayNum = compactRays(rayNum, devRayPool1, devRayPool2);
	compactionTime += tok();

	std::swap(devRayPool1, devRayPool2);
	return rayNum;
}

// delete rays that hit nothing
int PathTracer::compactRays(int rayNum, Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults) {
	thrust::device_ptr<Path> tRaysIn{ rayPool };
	thrust::device_ptr<Path> tRaysOut{ compactedRayPool };
	thrust::device_ptr<Path> tTerminated{ devTerminatedRays };
	thrust::device_ptr<IntersectInfo> tResultIn{ intersectResults };
	thrust::device_ptr<IntersectInfo> tResultOut{ compactedIntersectResults };

	thrust::device_ptr<Path> tmp = thrust::copy_if(tRaysIn, tRaysIn + rayNum, tRaysOut, ifHit());
	checkCUDAError("thrust::copy_if failed.");
	thrust::copy_if(tRaysIn, tRaysIn + rayNum, tTerminated + terminatedRayNum, ifNotHit());
	checkCUDAError("thrust::copy_if failed.");
	thrust::copy_if(tResultIn, tResultIn + rayNum, tRaysIn, tResultOut, ifHit());
	checkCUDAError("thrust::copy_if failed.");

	int remaining = tmp - tRaysOut;
	terminatedRayNum += (rayNum - remaining);
	return remaining;
}

// delete terminated rays
int PathTracer::compactRays(int rayNum, Path* rayPool, Path* compactedRayPool) {
	thrust::device_ptr<Path> tRaysIn{ rayPool };
	thrust::device_ptr<Path> tRaysOut{ compactedRayPool };
	thrust::device_ptr<Path> tTerminated{ devTerminatedRays };

	thrust::device_ptr<Path> tmp = thrust::copy_if(tRaysIn, tRaysIn + rayNum, tRaysOut, ifNotTerminated());
	thrust::copy_if(tRaysIn, tRaysIn + rayNum, tTerminated + terminatedRayNum, ifTerminated());
	checkCUDAError("thrust::copy_if failed.");

	int remaining = tmp - tRaysOut;
	terminatedRayNum += (rayNum - remaining);
	return remaining;
}

__global__ void kernInitializeRays(WindowSize window, int spp, Path* rayPool, int maxBounce, const Camera cam, bool jitter) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;
	thrust::default_random_engine rng = makeSeededRandomEngine(spp, idx, 0);
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);

	Path path{};
	path.pixelIdx = idx;
	path.remainingBounces = maxBounce;
	path.ray.origin = cam.position;

	int py = idx / window.width;
	int px = idx - py * window.width;

	float theta = TWO_PI * u01(rng);
	float r = u01(rng) * cam.apenture;

	float rnd1{ 0.f };
	float rnd2{ 0.f };

	if (jitter) {
		rnd1 = u01(rng) - 0.5f;
		rnd2 = u01(rng) - 0.5f;
	}

	glm::vec3 lookAt = cam.filmOrigin
		- cam.upDir * (((float)py + rnd1) * cam.pixelHeight + cam.halfPixelHeight)
		+ cam.rightDir * (((float)px + rnd2) * cam.pixelWidth + cam.halfPixelWidth);
	glm::vec3 offset = -cam.upDir * r * glm::cos(theta);
	offset += cam.rightDir * r * glm::sin(theta);

	path.ray.dir = glm::normalize(lookAt - cam.position - offset* CAMERA_MULTIPLIER);
	path.ray.invDir = 1.f / path.ray.dir;
	path.ray.origin = cam.position + offset;
	path.lastHit = 1;
	path.color = glm::vec3{ 1.f, 1.f, 1.f };
	rayPool[idx] = path;
}

__global__ void kernInitializeFrameBuffer(WindowSize window, float* frame) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;
	frame[idx * 3] = 0.f;
	frame[idx * 3 + 1] = 0.f;
	frame[idx * 3 + 2] = 0.f;
}

__global__ void kernWriteFrameBuffer(WindowSize window, float currentSpp, Path* rayPool, float* frameBuffer) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;

	Path path = rayPool[idx];
	if (path.lastHit < 0) { // ray didn't hit anything, or didn't hit light source in the end
		path.color = glm::vec3{ 0.f };
	}
	path.color = isinf(path.color.x) || isnan(path.color.x) ||
		isinf(path.color.y) || isnan(path.color.y) || 
		isinf(path.color.z) || isnan(path.color.z) ? glm::vec3{ 0.f } : path.color;
	frameBuffer[path.pixelIdx * 3] = (frameBuffer[path.pixelIdx * 3] * (currentSpp - 1.f) + path.color.x) / currentSpp;
	frameBuffer[path.pixelIdx * 3+1] = (frameBuffer[path.pixelIdx * 3+1] * (currentSpp - 1.f) + path.color.y) / currentSpp;
	frameBuffer[path.pixelIdx * 3+2] = (frameBuffer[path.pixelIdx * 3+2] * (currentSpp - 1.f) + path.color.z) / currentSpp;
}

void PathTracer::writeFrameBuffer(int spp) {
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernWriteFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>> (window, (float)spp, devTerminatedRays, devFrameBuf);

}


__global__ void kernShadeWithSkybox(int rayNum, cudaTextureObject_t skybox, glm::vec3 rotate, glm::vec3 up, glm::vec3 right, Path* rayPool) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	//todo

	Path p = rayPool[idx];
	if (p.lastHit < 0) {
		glm::vec3 dir = glm::normalize(p.ray.dir);
		vecTransform2(&dir, getRotationMat(rotate));

		float u = glm::fract(glm::atan(dir.z, dir.x) * INV_TWO_PI + 1.f);
		float v = glm::atan(glm::length(glm::vec2(dir.x, dir.z)), dir.y) * INV_PI;
		float4 texColor = tex2D<float4>(skybox, u, v);
		p.color *= glm::vec3{ texColor.x, texColor.y, texColor.z };

		p.lastHit = 1;
		p.remainingBounces = 0;

		rayPool[idx] = p;
	}
}

void PathTracer::shadeWithSkybox() {
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernShadeWithSkybox <<<blocksPerGrid, BLOCK_SIZE>>> (
		window.pixels, scene.skybox.devTexture, { 0.f, 0.f, 0.f }, scene.cam.upDir, { 1.f, 0.f, 0.f }, devTerminatedRays);
}

__global__ void kernGenerateSkyboxAlbedo(
	int rayNum, float currentSpp, cudaTextureObject_t skybox, glm::vec3 rotate, glm::vec3 up, glm::vec3 right, Path* rayPool, float* albedoBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Path p = rayPool[idx];
	if (p.lastHit < 0) {
		glm::vec3 dir = glm::normalize(p.ray.dir);
		vecTransform2(&dir, getRotationMat(rotate));
		float u = glm::fract(glm::atan(dir.z, dir.x) * INV_TWO_PI + 1.f);
		float v = glm::atan(glm::length(glm::vec2(dir.x, dir.z)), dir.y) * INV_PI;
		float4 texColor = tex2D<float4>(skybox, u, v);
		glm::vec3 albedo = glm::vec3{ texColor.x, texColor.y, texColor.z };
		albedoBuf[p.pixelIdx * 3] = (albedoBuf[p.pixelIdx * 3] * (currentSpp - 1.f) + albedo.x) / currentSpp;
		albedoBuf[p.pixelIdx * 3+1] = (albedoBuf[p.pixelIdx * 3 + 1] * (currentSpp - 1.f) + albedo.y) / currentSpp;
		albedoBuf[p.pixelIdx * 3 + 2] = (albedoBuf[p.pixelIdx * 3 + 2] * (currentSpp - 1.f) + albedo.z) / currentSpp;
	}
}
void PathTracer::generateSkyboxAlbedo(int rayNum, int spp) {
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernGenerateSkyboxAlbedo<<<blocksPerGrid, BLOCK_SIZE>>>(
		window.pixels, (float)spp, scene.skybox.devTexture, { 0.f, 0.f, 0.f }, scene.cam.upDir, { 1.f, 0.f, 0.f }, devTerminatedRays, devAlbedoBuf);
}

std::unique_ptr<float[]> PathTracer::getFrameBuffer() {
	if (devFrameBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaMemcpy(ptr.get(), devFrameBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Frame buffer isn't allocated yet.");
	}
}
void PathTracer::copyFrameBuffer(float* frameBuffer) {
	if (devFrameBuf) {
		cudaMemcpy(frameBuffer, devFrameBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	}
	else {
		throw std::runtime_error("Error: Frame buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getNormalBuffer() {
	if (devNormalBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaMemcpy(ptr.get(), devNormalBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Normal buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getAlbedoBuffer() {
	if (devAlbedoBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaMemcpy(ptr.get(), devAlbedoBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Albedo buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getDepthBuffer() {
	if (devDepthBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels] };
		cudaMemcpy(ptr.get(), devDepthBuf, window.pixels * sizeof(float), cudaMemcpyDeviceToHost);
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Depth buffer isn't allocated yet.");
	}
}

}