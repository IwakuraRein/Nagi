#include "path_tracer.cuh"

#include <thrust/host_vector.h>  
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

namespace nagi {
PathTracer::PathTracer(Scene& Scene, BVH& BVH) :scene{ Scene }, window{ scene.window }, bvh{ BVH }{
	destroyBuffers();
	allocateBuffers();

	cudaRun(cudaMemcpy(devObjBuf, scene.objBuf.data(), scene.objBuf.size() * sizeof(Object), cudaMemcpyHostToDevice));
	cudaRun(cudaMemcpy(devMtlBuf, scene.mtlBuf.data(), scene.mtlBuf.size() * sizeof(Material), cudaMemcpyHostToDevice));
	cudaRun(cudaMemcpy(devTrigBuf, scene.trigBuf.data(), scene.trigBuf.size() * sizeof(Triangle), cudaMemcpyHostToDevice));

	// no need to clear frame buffer. kernWriteFrameBuffer() will do this when spp=1
	//dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	//kernInitializeFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>>(window, devFrameBuf);
	//checkCUDAError("kernInitializeFrameBuffer failed.");

#ifdef DEB_INFO
	cudaRun(cudaEventCreate(&timer_start));
	cudaRun(cudaEventCreate(&timer_end));
#endif // DEB_INFO
}

void PathTracer::allocateBuffers() {
	cudaRun(cudaMalloc((void**)&devObjBuf, scene.objBuf.size() * sizeof(Object)));
	cudaRun(cudaMalloc((void**)&devMtlBuf, scene.mtlBuf.size() * sizeof(Material)));
	cudaRun(cudaMalloc((void**)&devTrigBuf, scene.trigBuf.size() * sizeof(Triangle)));
	cudaRun(cudaMalloc((void**)&devFrameBuf, sizeof(float) * window.pixels * 3));
	cudaRun(cudaMalloc((void**)&devNormalBuf, sizeof(float) * window.pixels * 3));
	cudaRun(cudaMalloc((void**)&devAlbedoBuf, sizeof(float) * window.pixels * 3));
	cudaRun(cudaMalloc((void**)&devDepthBuf, sizeof(float) * window.pixels));
	cudaRun(cudaMalloc((void**)&devCurrentNormalBuf, sizeof(float) * window.pixels * 3));
	cudaRun(cudaMalloc((void**)&devCurrentAlbedoBuf, sizeof(float) * window.pixels * 3));
	cudaRun(cudaMalloc((void**)&devCurrentDepthBuf, sizeof(float) * window.pixels));
	cudaRun(cudaMalloc((void**)&devLumiance2Buf, sizeof(float) * window.pixels));
	cudaRun(cudaMalloc((void**)&devRayPool1, window.pixels * sizeof(Path)));
	cudaRun(cudaMalloc((void**)&devRayPool2, window.pixels * sizeof(Path)));
	cudaRun(cudaMalloc((void**)&devTerminatedRays, window.pixels * sizeof(Path)));
	cudaRun(cudaMalloc((void**)&devResults1, window.pixels * sizeof(IntersectInfo)));
	cudaRun(cudaMalloc((void**)&devResults2, window.pixels * sizeof(IntersectInfo)));
}

void PathTracer::destroyBuffers() {
	if (devObjBuf) {
		cudaRun(cudaFree(devObjBuf));
		devObjBuf = nullptr;
	}
	if (devMtlBuf) {
		cudaRun(cudaFree(devMtlBuf));
		devMtlBuf = nullptr;
	}
	if (devTrigBuf) {
		cudaRun(cudaFree(devTrigBuf));
		devTrigBuf = nullptr;
	}
	if (devRayPool1) {
		cudaRun(cudaFree(devRayPool1));
		devRayPool1 = nullptr;
	}
	if (devRayPool2) {
		cudaRun(cudaFree(devRayPool2));
		devRayPool2 = nullptr;
	}
	if (devTerminatedRays) {
		cudaRun(cudaFree(devTerminatedRays));
		devTerminatedRays = nullptr;
	}
	if (devResults1) {
		cudaRun(cudaFree(devResults1));
		devResults1 = nullptr;
	}
	if (devResults2) {
		cudaRun(cudaFree(devResults2));
		devResults2 = nullptr;
	}
	if (devFrameBuf) {
		cudaRun(cudaFree(devFrameBuf));
		devFrameBuf = nullptr;
	}
	if (devNormalBuf) {
		cudaRun(cudaFree(devNormalBuf));
		devNormalBuf = nullptr;
	}
	if (devAlbedoBuf) {
		cudaRun(cudaFree(devAlbedoBuf));
		devAlbedoBuf = nullptr;
	}
	if (devDepthBuf) {
		cudaRun(cudaFree(devDepthBuf));
		devDepthBuf = nullptr;
	}
	if (devCurrentNormalBuf) {
		cudaRun(cudaFree(devCurrentNormalBuf));
		devCurrentNormalBuf = nullptr;
	}
	if (devCurrentAlbedoBuf) {
		cudaRun(cudaFree(devCurrentAlbedoBuf));
		devCurrentAlbedoBuf = nullptr;
	}
	if (devCurrentDepthBuf) {
		cudaRun(cudaFree(devCurrentDepthBuf));
		devCurrentDepthBuf = nullptr;
	}
	if (devLumiance2Buf) {
		cudaRun(cudaFree(devLumiance2Buf));
		devLumiance2Buf = nullptr;
	}
}

PathTracer::~PathTracer() {
	destroyBuffers();
#ifdef DEB_INFO
	if (timer_start)
		cudaRun(cudaEventDestroy(timer_start));
	if (timer_end)
		cudaRun(cudaEventDestroy(timer_end));
#endif // DEB_INFO
}

// intersection test -> compact rays -> sort rays according to material -> compute color -> compact rays -> intersection test...
void PathTracer::iterate() {
	std::chrono::steady_clock::time_point timer = std::chrono::high_resolution_clock::now();
#ifdef DEB_INFO
	std::cout << "  Begin iteration " << spp << ". " << scene.config.spp - spp << " remaining." << std::endl;
	float lastIntersectionTime = intersectionTime;
	float lastShadingTime = shadingTime;
	float lastCompactionTime = compactionTime;
	float lastGbufferTime = gbufferTime;
#endif // DEB_INFO

	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeRays<<<blocksPerGrid, BLOCK_SIZE>>> (window, spp, devRayPool1, scene.config.maxBounce, scene.cam, spp != 1);
	int bounce = 0;
	int remainingRays = window.pixels;
	while (true) {
		remainingRays = intersectionTest(remainingRays);
		bounce++;

		//std::cout << remainingRays << " ";
		if (remainingRays <= 0) break;

		//sortRays(remainingRays);

#ifdef DEB_INFO
		tik();
#endif // DEB_INFO
		if (bounce == 1 && scene.hasSkyBox) {
			generateSkyboxAlbedo(window.pixels - remainingRays, spp);
		}
		if (bounce <= MAX_GBUFFER_BOUNCE) {
			generateGbuffer(remainingRays, spp, bounce);
		}
#ifdef DEB_INFO
		gbufferTime += tok();
#endif // DEB_INFO

		remainingRays = shade(remainingRays, spp, bounce);
		//std::cout << remainingRays << std::endl;
		if (remainingRays <= 0) break;
	}
	if (scene.hasSkyBox) {
		shadeWithSkybox();
	}
	writeFrameBuffer(spp);
	terminatedRayNum = 0;
	spp++;

	float delta = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
#ifdef DEB_INFO
	std::cout << "    Intersection test: " << intersectionTime - lastIntersectionTime << " ms." << std::endl;
	std::cout << "    Shading: " << shadingTime - lastShadingTime << " ms." << std::endl;
	std::cout << "    Compaction : " << compactionTime - lastCompactionTime << " ms." << std::endl;
	std::cout << "    Gbuffer generation : " << gbufferTime - lastGbufferTime << " ms." << std::endl;
	std::cout << "  Iteration " << spp - 1 << " finished. Time cost: " << delta << " seconds." << std::endl;
	intersectionTime = 0.f;
	compactionTime = 0.f;
	shadingTime = 0.f;
	gbufferTime = 0.f;
#endif // DEB_INFO
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
	int rayNum, Path* rayPool, int treeSize, BVH::Node* treeBuf, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec2 pickedBaryCentric, baryCentric;
	Triangle pickedTrig;
	int pickedMtlIdx{ -1 };
	float dist;
	float minTrigD{ FLT_MAX };
	int nodeIdx = 0;
	while (nodeIdx < treeSize && nodeIdx >= 0) {
		BVH::Node node = treeBuf[nodeIdx];
		if (rayBoxIntersect(r, node.min, node.max, dist) && dist < minTrigD) {
			if (dist >= minTrigD && node.trigIdx < 0) nodeIdx = node.missLink;
			else {
				if (node.trigIdx >= 0) {
					Triangle trig = trigBuf[node.trigIdx];
					if (rayTrigIntersect(r, trig, dist, baryCentric)) {
						if (dist > 0.f && dist < minTrigD) {
							minTrigD = dist;
							pickedBaryCentric = baryCentric;
							pickedTrig = trig;
							pickedMtlIdx = trig.mtlIdx;
						}
					}
				}
				nodeIdx++;
			}
		}
		else nodeIdx = node.missLink;
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

#ifdef DEB_INFO
	tik();
#endif // DEB_INFO;
	//kernTrigIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, 0, scene.trigBuf.size()-1, devTrigBuf, devResults1);
	//kernObjIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, scene.objBuf.size(), devObjBuf, devTrigBuf, devResults1);
	kernBVHIntersectTest<<<blocksPerGrid, BLOCK_SIZE>>> (
		rayNum, devRayPool1, bvh.tree.size(), bvh.devTree, devTrigBuf, devResults1);
#ifdef DEB_INFO
	intersectionTime += tok();
#endif // DEB_INFO;

#ifdef DEB_INFO
	tik();
#endif // DEB_INFO;
	rayNum = compactRays(rayNum, devRayPool1, devRayPool2, devResults1, devResults2);
#ifdef DEB_INFO
	compactionTime += tok();
#endif // DEB_INFO;

	std::swap(devRayPool1, devRayPool2);
	std::swap(devResults1, devResults2);

	return rayNum;
}

// sort rays according to materials
void PathTracer::sortRays(int rayNum) {
	thrust::device_ptr<Path> tRays{ devRayPool1 };
	thrust::device_ptr<IntersectInfo> tResults{ devResults1 };

	thrust::stable_sort_by_key(tResults, tResults + rayNum, tRays, IntersectionComp());
	thrust::stable_sort(tResults, tResults + rayNum, IntersectionComp());
}

__global__ void kernGenerateGbuffer(
	int rayNum, float currentSpp, int bounce, glm::vec3 camPos, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf,
	float* currentAlbedoBuf, float* currentNormalBuf, float* currentDepthBuf, float* albedoBuf, float* normalBuf, float* depthBuf) {
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

	if (mtl.type == MTL_TYPE_LIGHT_SOURCE) {
		albedo = glm::vec3{ 1.f };
	}
	else if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
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

		// always record depth
		currentDepthBuf[pixel] = depth;

		if (p.type == PIXEL_TYPE_SPECULAR) { // nothing terminate
			currentAlbedoBuf[pixel * 3] = albedo.x;
			currentAlbedoBuf[pixel * 3 + 1] = albedo.y;
			currentAlbedoBuf[pixel * 3 + 2] = albedo.z;
			// store normal in case reach max gbuffer bounce
			currentNormalBuf[pixel * 3] = normal.x;
			currentNormalBuf[pixel * 3 + 1] = normal.y;
			currentNormalBuf[pixel * 3 + 2] = normal.z;
		}
		else if (p.type == PIXEL_TYPE_GLOSSY) { // depth and normal terminate, prepare to blend second bounce albedo
			currentAlbedoBuf[pixel * 3] = albedo.x;
			currentAlbedoBuf[pixel * 3 + 1] = albedo.y;
			currentAlbedoBuf[pixel * 3 + 2] = albedo.z;
			depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + depth) / currentSpp;

			currentNormalBuf[pixel * 3] = normal.x;
			currentNormalBuf[pixel * 3 + 1] = normal.y;
			currentNormalBuf[pixel * 3 + 2] = normal.z;
			normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
			normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
			normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
		}
		else {
			p.type == PIXEL_TYPE_DIFFUSE; // all terminate
			p.gbufferStored = true;
			albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + albedo.x) / currentSpp;
			albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + albedo.y) / currentSpp;
			albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + albedo.z) / currentSpp;

			depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + depth) / currentSpp;

			currentNormalBuf[pixel * 3] = normal.x;
			currentNormalBuf[pixel * 3 + 1] = normal.y;
			currentNormalBuf[pixel * 3 + 2] = normal.z;
			normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
			normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
			normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
		}

		rayPool[idx] = p;
	}

	else {
		float z = currentDepthBuf[idx] + depth;
		currentDepthBuf[idx] += depth;
		if (p.type == PIXEL_TYPE_GLOSSY && bounce == 2) { // blend second bounce albedo
			p.gbufferStored = true;
			rayPool[idx] = p;
			albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3] * albedo.x) / currentSpp;
			albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 1] * albedo.y) / currentSpp;
			albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 2] * albedo.z) / currentSpp;
		}
		if (p.type == PIXEL_TYPE_SPECULAR) {
			if (mtl.type != MTL_TYPE_SPECULAR || bounce == MAX_GBUFFER_BOUNCE) {
				p.gbufferStored = true;
				rayPool[idx] = p;
				albedoBuf[pixel * 3] = (albedoBuf[pixel * 3] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3] * albedo.x) / currentSpp;
				albedoBuf[pixel * 3 + 1] = (albedoBuf[pixel * 3 + 1] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 1] * albedo.y) / currentSpp;
				albedoBuf[pixel * 3 + 2] = (albedoBuf[pixel * 3 + 2] * (currentSpp - 1.f) + currentAlbedoBuf[pixel * 3 + 2] * albedo.z) / currentSpp;

				depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + z) / currentSpp;

				currentNormalBuf[pixel * 3] = normal.x;
				currentNormalBuf[pixel * 3 + 1] = normal.y;
				currentNormalBuf[pixel * 3 + 2] = normal.z;
				normalBuf[pixel * 3] = (normalBuf[pixel * 3] * (currentSpp - 1.f) + normal.x) / currentSpp;
				normalBuf[pixel * 3 + 1] = (normalBuf[pixel * 3 + 1] * (currentSpp - 1.f) + normal.y) / currentSpp;
				normalBuf[pixel * 3 + 2] = (normalBuf[pixel * 3 + 2] * (currentSpp - 1.f) + normal.z) / currentSpp;
			}
			else {
				currentAlbedoBuf[pixel * 3] *= albedo.x;
				currentAlbedoBuf[pixel * 3 + 1] *= albedo.y;
				currentAlbedoBuf[pixel * 3 + 2] *= albedo.z;
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
	kernGenerateGbuffer<<<blocksPerGrid, BLOCK_SIZE>>> (
		rayNum, (float)spp, bounce, scene.cam.position, devRayPool1, devResults1, devMtlBuf, devCurrentAlbedoBuf, devCurrentNormalBuf, devCurrentDepthBuf, devAlbedoBuf, devNormalBuf, devDepthBuf);
}


// compute color and generate new ray direction
__global__ void kernShadeLambert(int rayNum, int spp, int bounce, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
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
		auto rng = makeSeededRandomEngine(spp, idx, bounce);
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
		wo = cosHemisphereSampler(normal, pdf, rng);
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
__global__ void kernShadeSpecular(int rayNum, int spp, int bounce, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
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
		auto rng = makeSeededRandomEngine(spp, idx, bounce);
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
		glm::vec3 wo = reflectSampler(metallic, albedo, p.ray.dir, normal, pdf, rng);
		if (pdf < PDF_EPSILON) {
			p.color = glm::vec3{ 0.f };
			p.remainingBounces = 0;
		}
		else {
			glm::vec3 brdf = specularBrdf(p.ray.dir, wo, normal, albedo, metallic); // lambert is timed inside the bsdf
			p.color = p.color * brdf / pdf;
			p.ray.dir = wo;
			p.ray.invDir = 1.f / p.ray.dir;

			p.remainingBounces--;
		}
	}
	rayPool[idx] = p;
}
__global__ void kernShadeLightSource(int rayNum, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
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
__global__ void kernShadeGlass(int rayNum, int spp, int bounce, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
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
		auto rng = makeSeededRandomEngine(spp, idx, bounce);
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

		float pdf;
		wo = refractSampler(mtl.ior, p.ray.dir, normal, pdf, rng);
		p.color = p.color * albedo / pdf;
		p.ray.origin += wo * REFRACT_OFFSET;
		p.ray.dir = wo;
		p.ray.invDir = 1.f / wo;
		p.remainingBounces--;
	}
	rayPool[idx] = p;
}
__global__ void kernShadeMicrofacet(int rayNum, int spp, int bounce, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
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
		auto rng = makeSeededRandomEngine(spp, idx, bounce);
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

		wo = ggxImportanceSampler(roughness, metallic, p.ray.dir, normal, pdf, rng);

		if (glm::dot(wo, normal) < 0.f || pdf < PDF_EPSILON) {
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

int PathTracer::shade(int rayNum, int spp, int bounce) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);

#ifdef DEB_INFO
	tik();
#endif // DEB_INFO;
	if (hasMaterial(scene, MTL_TYPE_LIGHT_SOURCE))
		kernShadeLightSource<<<blocksPerGrid, BLOCK_SIZE>>> (rayNum, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_LAMBERT))
		kernShadeLambert<<<blocksPerGrid, BLOCK_SIZE>>> (rayNum, spp, bounce, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_SPECULAR))
		kernShadeSpecular<<<blocksPerGrid, BLOCK_SIZE>>> (rayNum, spp, bounce, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_GLASS))
		kernShadeGlass<<<blocksPerGrid, BLOCK_SIZE>>> (rayNum, spp, bounce, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_MICROFACET))
		kernShadeMicrofacet<<<blocksPerGrid, BLOCK_SIZE>>> (rayNum, spp, bounce, devRayPool1, devResults1, devMtlBuf);
#ifdef DEB_INFO
	shadingTime += tok();
#endif // DEB_INFO;

#ifdef DEB_INFO
	tik();
#endif // DEB_INFO;
	rayNum = compactRays(rayNum, devRayPool1, devRayPool2);
#ifdef DEB_INFO
	compactionTime += tok();
#endif // DEB_INFO;

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
	thrust::copy_if(tRaysIn, tRaysIn + rayNum, tTerminated + terminatedRayNum, ifNotHit());
	thrust::copy_if(tResultIn, tResultIn + rayNum, tRaysIn, tResultOut, ifHit());

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

	path.ray.dir = glm::normalize(lookAt - cam.position - offset * CAMERA_MULTIPLIER);
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

__global__ void kernWriteFrameBuffer(WindowSize window, float currentSpp, Path* rayPool, float* albedoBuffer, float* frameBuffer, float* luminanceBuffer) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;

	Path path = rayPool[idx];
	glm::vec3 color{ frameBuffer[path.pixelIdx * 3], frameBuffer[path.pixelIdx * 3 + 1], frameBuffer[path.pixelIdx * 3 + 2] };
	glm::vec3 albedo{ albedoBuffer[path.pixelIdx * 3], albedoBuffer[path.pixelIdx * 3 + 1], albedoBuffer[path.pixelIdx * 3 + 2] };
	float Y2{ luminanceBuffer[path.pixelIdx] };
	if (path.lastHit < 0) { // ray didn't hit anything, or didn't hit light source in the end
		path.color = glm::vec3{ 0.f };
	}
	path.color = isinf(path.color.x) || isnan(path.color.x) ||
		isinf(path.color.y) || isnan(path.color.y) ||
		isinf(path.color.z) || isnan(path.color.z) ? glm::vec3{ 0.f } : path.color;

	color = (color * (currentSpp - 1.f) + path.color) / currentSpp;
	glm::vec3 luminance = color / (albedo + FLT_EPSILON);
	float Y = 0.299f * luminance.x + 0.587f * luminance.y + 0.114f * luminance.z;
	Y2 = (Y2 * (currentSpp - 1.f) + Y * Y) / currentSpp;

	frameBuffer[path.pixelIdx * 3] = color.x;
	frameBuffer[path.pixelIdx * 3 + 1] = color.y;
	frameBuffer[path.pixelIdx * 3 + 2] = color.z;
	luminanceBuffer[path.pixelIdx] = Y2;
}

void PathTracer::writeFrameBuffer(int spp) {
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernWriteFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>> (window, (float)spp, devTerminatedRays, devAlbedoBuf, devFrameBuf, devLumiance2Buf);

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
	kernShadeWithSkybox<<<blocksPerGrid, BLOCK_SIZE>>> (
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
		albedoBuf[p.pixelIdx * 3 + 1] = (albedoBuf[p.pixelIdx * 3 + 1] * (currentSpp - 1.f) + albedo.y) / currentSpp;
		albedoBuf[p.pixelIdx * 3 + 2] = (albedoBuf[p.pixelIdx * 3 + 2] * (currentSpp - 1.f) + albedo.z) / currentSpp;
	}
}
void PathTracer::generateSkyboxAlbedo(int rayNum, int spp) {
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernGenerateSkyboxAlbedo<<<blocksPerGrid, BLOCK_SIZE>>> (
		window.pixels, (float)spp, scene.skybox.devTexture, { 0.f, 0.f, 0.f }, scene.cam.upDir, { 1.f, 0.f, 0.f }, devTerminatedRays, devAlbedoBuf);
}

std::unique_ptr<float[]> PathTracer::getFrameBuffer() {
	if (devFrameBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaRun(cudaMemcpy(ptr.get(), devFrameBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Frame buffer isn't allocated yet.");
	}
}
void PathTracer::copyFrameBuffer(float* frameBuffer) {
	if (devFrameBuf) {
		cudaRun(cudaMemcpy(frameBuffer, devFrameBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
	}
	else {
		throw std::runtime_error("Error: Frame buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getNormalBuffer() {
	if (devNormalBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaRun(cudaMemcpy(ptr.get(), devNormalBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Normal buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getAlbedoBuffer() {
	if (devAlbedoBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels * 3] };
		cudaRun(cudaMemcpy(ptr.get(), devAlbedoBuf, window.pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Albedo buffer isn't allocated yet.");
	}
}
std::unique_ptr<float[]> PathTracer::getDepthBuffer() {
	if (devDepthBuf) {
		std::unique_ptr<float[]> ptr{ new float[window.pixels] };
		cudaRun(cudaMemcpy(ptr.get(), devDepthBuf, window.pixels * sizeof(float), cudaMemcpyDeviceToHost));
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Depth buffer isn't allocated yet.");
	}
}

}