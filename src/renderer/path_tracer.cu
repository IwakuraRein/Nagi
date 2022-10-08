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
}

PathTracer::~PathTracer() {
	destroyBuffers();
}

// intersection test -> compact rays -> sort rays according to material -> compute color -> compact rays -> intersection test...
void PathTracer::iterate() {
	std::chrono::steady_clock::time_point timer;
	std::cout << "  Begin iteration " << spp << ". " << scene.config.spp - spp << " remaining." << std::endl;
	if (printDetails) timer = std::chrono::high_resolution_clock::now();
	dim3 blocksPerGrid((window.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeRays<<<blocksPerGrid, BLOCK_SIZE>>> (window, spp, devRayPool1, scene.config.maxBounce, scene.cam);
	checkCUDAError("kernInitializeRays failed.");
	bool firstIntersection = true;
	int remainingRays = window.pixels;
	while (true) {
		remainingRays = intersectionTest(remainingRays);

		//std::cout << remainingRays << " ";
		if (remainingRays <= 0) break;

		//sortRays(remainingRays);

		if (firstIntersection) {
			generateGbuffer(remainingRays, spp);
			if (scene.hasSkyBox) {
				generateGbufferWithSkybox();
			}
			firstIntersection = false;
		}

		remainingRays = shade(remainingRays, spp);
		//std::cout << remainingRays << std::endl;
		if (remainingRays <= 0) break;
	}
	//if (scene.hasSkyBox) {
	//	shadeWithSkybox();
	//}
	writeFrameBuffer(spp);
	terminatedRayNum = 0;
	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "  Iteration " << spp << " finished. Time cost: " << runningTime <<
		" seconds. Time Remaining: " << runningTime * (scene.config.spp - spp) << " seconds." << std::endl;
	spp++;
}

__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec3 normal, position, tangent;
	glm::vec3 pickedNormal{ 0.f, 0.f, 0.f };
	glm::vec3 pickedTangent{ 0.f, 0.f, 0.f };
	glm::vec2 pickedUV{ 0.f, 0.f };
	glm::vec2 uv;
	int pickedMtlIdx{ -1 };
	float dist;
	float minDist{ FLT_MAX };
	for (int i = trigIdxStart; i <= trigIdxEnd; i++) {
		Triangle trig = trigBuf[i];
		if (rayBoxIntersect(r, trig.bbox, &dist)) {
			if (rayTrigIntersect(r, trig, &dist, &normal, &tangent, &uv)) {
				if (dist > 0.f && dist < minDist) {
					minDist = dist;
					pickedNormal = normal;
					pickedTangent = tangent;
					pickedUV = uv;
					pickedMtlIdx = trig.mtlIdx;
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = pickedNormal;
	result.tangent = pickedTangent;
	result.uv = pickedUV;
	//result.position = r.origin + r.dir * (minDist - 0.001f);
	r.origin = r.origin + r.dir * (minDist - 0.001f);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;
}

__global__ void kernObjIntersectTest(int rayNum, Path* rayPool, int objNum, Object* objBuf, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec3 normal, position, tangent;
	glm::vec3 pickedNormal{ 0.f, 0.f, 0.f };
	glm::vec3 pickedTangent{ 0.f, 0.f, 0.f };
	glm::vec2 pickedUV{ 0.f, 0.f };
	glm::vec2 uv;
	int pickedMtlIdx{ -1 };
	float dist;
	float minDist{ FLT_MAX };
	for (int i = 0; i < objNum; i++) {
		Object obj = objBuf[i];
		if (rayBoxIntersect(r, obj.bbox, &dist)) {
			if (dist < minDist) {
				for (int j = obj.trigIdxStart; j <= obj.trigIdxEnd; j++) {
					Triangle trig = trigBuf[j];
					if (rayBoxIntersect(r, trig.bbox, &dist)) {
						if (rayTrigIntersect(r, trig, &dist, &normal, &tangent, &uv)) {
							if (dist > 0.f && dist < minDist) {
								minDist = dist;
								pickedNormal = normal;
								pickedUV = uv;
								pickedMtlIdx = trig.mtlIdx;
								pickedTangent = tangent;
							}
						}
					}
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = pickedNormal;
	result.tangent = pickedTangent;
	result.uv = pickedUV;
	result.position = r.origin + r.dir * (minDist - 0.001f);
	//r.origin = r.origin + r.dir * (minDist - 0.001f);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;

}

__global__ void kernBVHIntersectTest(
	int rayNum, Path* rayPool, int objNum, Object* objBuf, BVH::Node* treeBuf, int* treeTrigIdx, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec3 normal, position, tangent;
	glm::vec3 pickedNormal{ 0.f };
	glm::vec3 pickedTangent{ 0.f };
	glm::vec2 pickedUV{ 0.f };
	glm::vec2 uv;
	int pickedMtlIdx{ -1 };
	float trigDist;
	float minTrigD{ FLT_MAX };
	float boxD;
	//BoundingBox lastTrigBox{};

	BVH::Node stack[MAX_TREE_DEPTH + 1];
	int searchedChildern[MAX_TREE_DEPTH + 1] = { 0 };
	for (int i = 0; i < objNum; i++) {
		Object obj = objBuf[i];
		if (rayBoxIntersect(r, obj.bbox, &boxD)) {
			if (boxD < minTrigD) {
				int ptr = 0;
				stack[0] = treeBuf[obj.treeRoot]; // root node;
				while (ptr >= 0) {
					BVH::Node& node = stack[ptr];
					if (node.trigIdxStart < 0) { // not leaf node
						if (node.size > searchedChildern[ptr]) {
							int childNo = searchedChildern[ptr];
							searchedChildern[ptr]++;
							if (rayBoxIntersect(r, node.childrenMin[childNo], node.childrenMax[childNo], &boxD)) {
								if (boxD < minTrigD) {
									ptr++;
									stack[ptr] = treeBuf[node.children[childNo]];
									searchedChildern[ptr] = 0;
									continue;
								}
							}
						}
						else {
							searchedChildern[ptr] = 0;
							ptr--;
						}
					}
					else {
						for (int i = node.trigIdxStart; i < node.trigIdxStart + node.size; i++) {
							Triangle trig = trigBuf[treeTrigIdx[i]];
							if (rayBoxIntersect(r, trig.bbox, &trigDist)) {
								//if (trigDist >= minTrigD) {
								//	if (!boxBoxIntersect(trig.bbox, lastTrigBox)) continue;
								//}
								if (rayTrigIntersect(r, trig, &trigDist, &normal, &tangent, &uv)) {
									if (trigDist > 0.f && trigDist < minTrigD) {
										minTrigD = trigDist;
										pickedNormal = normal;
										pickedUV = uv;
										pickedMtlIdx = trig.mtlIdx;
										pickedTangent = tangent;
										//lastTrigBox = trig.bbox;
									}
								}
							}
						}
						searchedChildern[ptr] = 0;
						ptr--;
					}
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = pickedNormal;
	result.tangent = pickedTangent;
	result.uv = pickedUV;
	result.position = r.origin + r.dir * (minTrigD - 0.001f);
	//r.origin = r.origin + r.dir * (minTrigD - 0.001f);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;
}

int PathTracer::intersectionTest(int rayNum) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
	//kernTrigIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, 0, scene.trigBuf.size()-1, devTrigBuf, devResults1);
	//kernObjIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, scene.objBuf.size(), devObjBuf, devTrigBuf, devResults1);
	kernBVHIntersectTest<<<blocksPerGrid, BLOCK_SIZE>>> (
		rayNum, devRayPool1, scene.objBuf.size(), devObjBuf, bvh.devTree, bvh.devTreeTrigIdx, devTrigBuf, devResults1);
	checkCUDAError("kernBVHIntersectTest failed.");

	rayNum = compactRays(rayNum, devRayPool1, devRayPool2, devResults1, devResults2);

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
	int rayNum, float currentSpp, glm::vec3 camPos, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf, float* albedoBuf, float* normalBuf, float* depthBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Path p = rayPool[idx];
	int pixel = p.pixelIdx;
	IntersectInfo intersect = intersections[idx];
	Material mtl = mtlBuf[intersect.mtlIdx];

	glm::vec3 normal;
	if (hasTexture(mtl, TEXTURE_TYPE_NORMAL)) {
		glm::mat3 TBN = glm::mat3(intersect.tangent, glm::cross(intersect.normal, intersect.tangent), intersect.normal);
		float4 texVal = tex2D<float4>(mtl.normalTex.devTexture, intersect.uv.x, intersect.uv.y);
		glm::vec3 bump{ texVal.x * 2.f - 1.f, texVal.y * 2.f - 1.f, 1.f };
		bump.z = sqrtf(1.f - glm::clamp(bump.x * bump.x + bump.y * bump.y, 0.f, 1.f));
		normal = glm::normalize(TBN * bump);
	}
	else normal = intersect.normal;

	glm::vec3 albedo;
	if (hasTexture(mtl, TEXTURE_TYPE_BASE)) {
		float4 baseTex = tex2D<float4>(mtl.baseTex.devTexture, intersect.uv.x, intersect.uv.y);
		albedo = glm::vec3{ baseTex.x, baseTex.y, baseTex.z };
	}
	else albedo = mtl.albedo;

	float depth = glm::length(intersect.position - camPos);

	// blend the gbuffer is good for denoising. 
	// reference: https://github.com/tunabrain/tungsten/issues/69
	normalBuf[pixel * 3] = (normalBuf[pixel * 3]*(currentSpp - 1.f) + normal.x) / currentSpp;
	normalBuf[pixel * 3+1] = (normalBuf[pixel * 3+1]*(currentSpp - 1.f) + normal.y) / currentSpp;
	normalBuf[pixel * 3+2] = (normalBuf[pixel * 3+2]*(currentSpp - 1.f) + normal.z) / currentSpp;
	albedoBuf[pixel * 3] = (albedoBuf[pixel * 3]*(currentSpp - 1.f) + albedo.x) / currentSpp;
	albedoBuf[pixel * 3+1] = (albedoBuf[pixel * 3+1]*(currentSpp - 1.f) + albedo.y) / currentSpp;
	albedoBuf[pixel * 3+2] = (albedoBuf[pixel * 3+2]*(currentSpp - 1.f) + albedo.z) / currentSpp;
	depthBuf[pixel] = (depthBuf[pixel] * (currentSpp - 1.f) + depth) / currentSpp;
}
void PathTracer::generateGbuffer(int rayNum, int spp) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernGenerateGbuffer<<<blocksPerGrid, BLOCK_SIZE >> > (
		rayNum, (float)spp, scene.cam.position, devRayPool1, devResults1, devMtlBuf, devAlbedoBuf, devNormalBuf, devDepthBuf);
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
		wo = cosHemisphereSampler(normal, &pdf, samplingRnd1, samplingRnd2);
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
__global__ void kernShadeMirror(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];
	if (mtl.type != MTL_TYPE_MIRROR) return;

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
		glm::vec3 normal, albedo;
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
		p.color = p.color * albedo; // lambert is timed inside the bsdf
		p.ray.dir = glm::reflect(p.ray.dir, normal);
		p.ray.invDir = 1.f / p.ray.dir;

		p.remainingBounces--;
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
		float f;
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

		p.color = p.color * albedo;

		bool enter{ true };
		if (glm::dot(p.ray.dir, normal) > 0.f) {
			enter = false;
			normal = -normal;
		}
		//float aaa = glm::dot(p.ray.dir, normal);
		wo = refractionSampler(mtl.ior, p.ray.dir, normal, &f, &enter);
		//float bbb = glm::dot(p.ray.dir, wo);
		//printf("%f %f %f %f\n", mtl.ior, aaa, bbb, f);
		float r = u01(rng);
		if (r < f) { //reflect
			wo = glm::reflect(p.ray.dir, normal);
		}
		p.ray.origin += wo * 0.002f;
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
		wo = ggxImportanceSampler(roughness, p.ray.dir, normal, &pdf, samplingRnd1, samplingRnd2);
		glm::vec3 wo2 = cosHemisphereSampler(normal, &pdf2, samplingRnd1, samplingRnd2);
		if (r > s) wo = wo2;
		pdf = s * pdf + (1.f - s) * pdf2;

		if (glm::dot(wo, normal) <= 0.001f || pdf < PDF_EPSILON) {
			p.color = glm::vec3{ 0.f };
			p.remainingBounces = 0;
		}
		else {
			bsdf = microFacetBrdf(p.ray.dir, wo, normal, albedo, metallic, roughness);
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
	if (hasMaterial(scene, MTL_TYPE_LIGHT_SOURCE))
		kernShadeLightSource<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_LAMBERT))
		kernShadeLambert<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_MIRROR))
		kernShadeMirror<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_GLASS))
		kernShadeGlass<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	if (hasMaterial(scene, MTL_TYPE_MICROFACET))
		kernShadeMicrofacet<<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);

	checkCUDAError("kernShading failed.");

	rayNum = compactRays(rayNum, devRayPool1, devRayPool2);
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

__global__ void kernInitializeRays(WindowSize window, int spp, Path* rayPool, int maxBounce, const Camera cam/*, bool jitter, bool DOP*/) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= window.pixels) return;
	thrust::default_random_engine rng = makeSeededRandomEngine(spp, idx, 0);
	thrust::uniform_real_distribution<double> u01(0.f, 1.f);

	Path path;
	path.pixelIdx = idx;
	path.remainingBounces = maxBounce;
	path.ray.origin = cam.position;

	int py = idx / window.width;
	int px = idx - py * window.width;

	//glm::vec3 ndc{ -1.f + px * PIXEL_WIDTH + HALF_PIXEL_WIDTH, -1.f + py * PIXEL_HEIGHT + HALF_PIXEL_HEIGHT, 0.5f };
	//vecTransform(&ndc, cam.invProjectMat*cam.invViewMat);
	//glm::vec3 dir = ndc - cam.position;

	float theta = TWO_PI * u01(rng);
	float r = u01(rng) * cam.apenture;

	float rnd1 = u01(rng) - 0.5f;
	float rnd2 = u01(rng) - 0.5f;

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

		//float theta = glm::acos(glm::dot(dir, up));
		float cosPhi = glm::dot(dir, right);
		float sinPhi = glm::length(glm::cross(dir, right));

		//float v = glm::clamp(theta * INV_PI, 0.f, 1.f);
		//float u = glm::clamp(0.5f+phi * INV_PI, 0.f, 1.f);
		float v = glm::dot(dir, up);
		float u;
		if (sinPhi > 0) {
			u = 0.5f * cosPhi;
		}
		else u = 1.f - 0.5f * cosPhi;
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

__global__ void kernGenerateGbufferWithSkybox(
	int rayNum, float currentSpp, cudaTextureObject_t skybox, glm::vec3 up, glm::vec3 right, Path* rayPool, float* albedoBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;
}
void PathTracer::generateGbufferWithSkybox() {
	//todo
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