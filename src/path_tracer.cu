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

	dim3 blocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuf);
	checkCUDAError("kernInitializeFrameBuffer failed.");
}

void PathTracer::allocateBuffers() {
	cudaMalloc((void**)&devObjBuf, scene.objBuf.size() * sizeof(Object));
	checkCUDAError("cudaMalloc devObjBuf failed.");
	cudaMalloc((void**)&devMtlBuf, scene.mtlBuf.size() * sizeof(Material));
	checkCUDAError("cudaMalloc devMtlBuf failed.");
	cudaMalloc((void**)&devTrigBuf, scene.trigBuf.size() * sizeof(Triangle));
	checkCUDAError("cudaMalloc devTrigBuf failed.");
	cudaMalloc((void**)&devFrameBuf, sizeof(float) * PIXEL_COUNT * 3);
	checkCUDAError("cudaMalloc devFrameBuf failed.");
	cudaMalloc((void**)&devRayPool1, PIXEL_COUNT * sizeof(Path));
	checkCUDAError("cudaMalloc devRayPool1 failed.");
	cudaMalloc((void**)&devRayPool2, PIXEL_COUNT * sizeof(Path));
	checkCUDAError("cudaMalloc devRayPool2 failed.");
	cudaMalloc((void**)&devTerminatedRays, PIXEL_COUNT * sizeof(Path));
	checkCUDAError("cudaMalloc devTerminatedRays failed.");
	cudaMalloc((void**)&devResults1, PIXEL_COUNT * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults1 failed.");
	cudaMalloc((void**)&devResults2, PIXEL_COUNT * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults2 failed.");
}

void PathTracer::destroyBuffers() {
	if (devObjBuf) {
		cudaFree(devObjBuf);
		checkCUDAError2("cudaFree devObjBuf failed.");
		devObjBuf = nullptr;
	}
	if (devMtlBuf) {
		cudaFree(devMtlBuf);
		checkCUDAError2("cudaFree devMtlBuf failed.");
		devMtlBuf = nullptr;
	}
	if (devTrigBuf) {
		cudaFree(devTrigBuf);
		checkCUDAError2("cudaFree devTrigBuf failed.");
		devTrigBuf = nullptr;
	}
	if (devRayPool1) {
		cudaFree(devRayPool1);
		checkCUDAError2("cudaFree devRayPool1 failed.");
		devRayPool1 = nullptr;
	}
	if (devRayPool2) {
		cudaFree(devRayPool2);
		checkCUDAError2("cudaFree devRayPool2 failed.");
		devRayPool2 = nullptr;
	}
	if (devTerminatedRays) {
		cudaFree(devTerminatedRays);
		checkCUDAError2("cudaFree v failed.");
		devTerminatedRays = nullptr;
	}
	if (devResults1) {
		cudaFree(devResults1);
		checkCUDAError2("cudaFree devResults1 failed.");
		devResults1 = nullptr;
	}
	if (devResults2) {
		cudaFree(devResults2);
		checkCUDAError2("cudaFree devResults2 failed.");
		devResults2 = nullptr;
	}
	if (devFrameBuf) {
		cudaFree(devFrameBuf);
		checkCUDAError2("cudaFree devFrameBuf failed.");
		devFrameBuf = nullptr;
	}
}

PathTracer::~PathTracer() {
	destroyBuffers();
}

// intersection test -> compact rays -> sort rays according to material -> compute color -> compact rays -> intersection test...
void PathTracer::iterate() {
	std::cout << "Start ray tracing..." << std::endl;

	std::chrono::steady_clock::time_point timer1, timer2;
	timer1 = std::chrono::high_resolution_clock::now();
	for (int spp = 1; spp <= scene.config.spp; spp++) {
		std::cout << "  Begin iteration " << spp << ". " << scene.config.spp - spp << " remaining." << std::endl;
		if (printDetails) timer2 = std::chrono::high_resolution_clock::now();

		dim3 blocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernInitializeRays<<<blocksPerGrid, BLOCK_SIZE>>>(spp, devRayPool1, scene.config.maxBounce, scene.cam);
		checkCUDAError("kernInitializeRays failed.");

		int remainingRays = PIXEL_COUNT;
		while (true) {
			remainingRays = intersectionTest(remainingRays);
			if (remainingRays <= 0) break;
			//std::cout << "    " << remainingRays << "; ";
			sortRays(remainingRays);

			remainingRays = shade(remainingRays, spp);
			if (remainingRays <= 0) break;
		}
		writeFrameBuffer();
		terminatedRayNum = 0;
		if (printDetails) {
			float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer2).count();
			std::cout << "  Iteration " << spp << " finished. Time cost: " << runningTime << " seconds." << std::endl;
		}
	}
	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer1).count();
	std::cout << "Ray tracing finished. Time cost: " << runningTime << " seconds." << std::endl;
}

__global__ void kernTrigIntersectTest(int rayNum, Path* rayPool, int trigIdxStart, int trigIdxEnd, Triangle* trigBuf, IntersectInfo* out) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Ray r = rayPool[idx].ray;
	glm::vec3 normal, position;
	glm::vec3 pickedNormal{ 0.f, 0.f, 0.f };
	int pickedMtlIdx{ -1 };
	float dist;
	float minDist{ FLT_MAX };
	for (int i = trigIdxStart; i < trigIdxEnd; i++) {
		Triangle trig = trigBuf[i];
		if (rayBoxIntersect(r, trig.bbox, &dist)) {
			if (rayTrigIntersect(r, trig, &dist, &normal)) {
				if (dist > 0.f && dist < minDist) {
					minDist = dist;
					pickedNormal = normal;
					pickedMtlIdx = trig.mtlIdx;
				}
			}
		}
	}
	IntersectInfo result;
	result.mtlIdx = pickedMtlIdx;
	result.normal = pickedNormal;
	result.position = r.origin + r.dir * (minDist - 0.001f);
	rayPool[idx].lastHit = pickedMtlIdx; // if pickedMtlIdx >=0, ray hits something
	out[idx] = result;
}

int PathTracer::intersectionTest(int rayNum) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernTrigIntersectTest <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, devRayPool1, 0, scene.trigBuf.size(), devTrigBuf, devResults1);
	checkCUDAError("kernTrigIntersectTest failed.");

	rayNum = compactRays(rayNum, devRayPool1, devRayPool2, devResults1, devResults2);

	std::swap(devRayPool1, devRayPool2);
	std::swap(devResults1, devResults2);

	return rayNum;
}

// sort rays according to materials
void PathTracer::sortRays(int rayNum) {
	thrust::device_ptr<Path> tRays{ devRayPool1 };
	thrust::device_ptr<IntersectInfo> tResults{ devResults1 };

	thrust::stable_sort_by_key(tResults, tResults+rayNum, tRays, IntersectionComp());
	checkCUDAError("thrust::stable_sort_by_key failed.");
	thrust::stable_sort(tResults, tResults+rayNum, IntersectionComp());
	checkCUDAError("thrust::stable_sort failed.");
}

// compute color and generate new ray direction
__global__ void kernShading(int rayNum, int spp, Path* rayPool, IntersectInfo* intersections, Material* mtlBuf) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= rayNum) return;

	Path p = rayPool[idx];
	IntersectInfo intersection = intersections[idx];
	Material mtl = mtlBuf[intersection.mtlIdx];

	p.ray.origin = intersection.position;

	if (mtl.type == MTL_TYPE_LIGHT_SOURCE) {
		p.color = p.color * mtl.emission;
		p.remainingBounces = 0;
	}
	else {
		if (p.remainingBounces == 0) {
			p.lastHit = -1;
		}
		else {
			float pdf;
			glm::vec3 wo = cosHemisphereSampler(intersection.normal, &pdf, makeSeededRandomEngine(spp, idx, 0));
			glm::vec3 bsdf = opaqueBsdf(p.ray.dir, wo, intersection.normal, mtl);
			p.color = p.color * bsdf + mtl.emission;
			//p.color = (wo + 1.f) / 2.f;
			//p.color = (intersection.normal + 1.f) / 2.f;
			p.ray.dir = wo;
			p.ray.invDir = 1.f / wo;
			p.remainingBounces--;
		}
	}
	rayPool[idx] = p;
}

int PathTracer::shade(int rayNum, int spp) {
	dim3 blocksPerGrid((rayNum + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernShading <<<blocksPerGrid, BLOCK_SIZE>>>(rayNum, spp, devRayPool1, devResults1, devMtlBuf);
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
	//return 0;
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
	//return 0;
}

//__global__ void kernInitializeRays(Path* rayPool, int maxBounce, const glm::vec3 camPos, const glm::mat4 invProjViewMat) {
__global__ void kernInitializeRays(int spp, Path* rayPool, int maxBounce, const Camera cam) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;

	thrust::default_random_engine rng = makeSeededRandomEngine(spp, idx, 0);

	thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
	float rnd1 = u01(rng);
	float rnd2 = u01(rng);

	Path path;
	path.pixelIdx = idx;
	path.remainingBounces = maxBounce;
	path.ray.origin = cam.position;

	int py = idx / WINDOW_WIDTH;
	int px = idx - py * WINDOW_HEIGHT;

	//todo: add jitter
	//glm::vec3 ndc{ -1.f + px * PIXEL_WIDTH + HALF_PIXEL_WIDTH, -1.f + py * PIXEL_HEIGHT + HALF_PIXEL_HEIGHT, 0.5f };
	//vecTransform(&ndc, cam.invProjectMat*cam.invViewMat);
	//glm::vec3 dir = ndc - cam.position;
	float halfh = tan(cam.fov / 2);
	float halfw = halfh * cam.aspect;
	path.ray.dir = cam.screenOrigin
		- cam.upDir * (((float)py + rnd1) * cam.pixelHeight + cam.pixelHeight / 2)
		+ cam.rightDir * (((float)px + rnd2) * cam.pixelWidth + cam.pixelWidth / 2);
	path.ray.dir = glm::normalize(path.ray.dir);
	path.ray.invDir = 1.f / path.ray.dir;
	path.lastHit = 1;
	path.color = glm::vec3{ 1.f, 1.f, 1.f };
	rayPool[idx] = path;
}

__global__ void kernInitializeFrameBuffer(float* frame) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;
	frame[idx * 3] = 0.f;
	frame[idx * 3 + 1] = 0.f;
	frame[idx * 3 + 2] = 0.f;
}

__global__ void kernWriteFrameBuffer(float spp, Path* rayPool, float* frameBuffer) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;

	Path path = rayPool[idx];
	if (path.lastHit < 0) { // ray didn't hit anything, or didn't hit light source in the end
		path.color = glm::vec3{ 0.f };
	}
	frameBuffer[path.pixelIdx * 3] += (path.color.x / spp);
	frameBuffer[path.pixelIdx * 3 + 1] += (path.color.y / spp);
	frameBuffer[path.pixelIdx * 3 + 2] += (path.color.z / spp);
}

void nagi::PathTracer::writeFrameBuffer() {
	dim3 blocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernWriteFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(scene.config.spp, devTerminatedRays, devFrameBuf);

}

//__global__ void kernTest(Path* rayPool, Triangle* trigBuf, int num, float* frameBuffer) {
//	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
//	if (idx >= PIXEL_COUNT) return;
//	Ray r = rayPool[idx].ray;
//	glm::vec3 normal;
//	glm::vec3 pickedNormal{ 0.f, 0.f, 0.f };
//	float dist;
//	float minDist{ FLT_MAX };
//	bool found{ false };
//	for (int i = 0; i < num; i++) {
//		Triangle trig = trigBuf[i];
//		if (rayBoxIntersect(r, trig.bbox, &dist)) {
//			if (rayTrigIntersect(r, trig, &dist, &normal)) {
//				if (dist < minDist) {
//					minDist = dist;
//					pickedNormal = normal;
//					found = true;
//				}
//			}
//		}
//	}
//	if (found) {
//		frameBuffer[idx * 3] = (pickedNormal.x + 1.f) / 2.f;
//		frameBuffer[idx * 3 + 1] = (pickedNormal.y + 1.f) / 2.f;
//		frameBuffer[idx * 3 + 2] = (pickedNormal.z + 1.f) / 2.f;
//	}
//}

std::unique_ptr<float[]> nagi::PathTracer::getFrameBuffer() {
	if (devFrameBuf) {
		std::unique_ptr<float[]> ptr{ new float[PIXEL_COUNT * 3] };
		cudaMemcpy(ptr.get(), devFrameBuf, PIXEL_COUNT * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		return ptr;
	}
	else {
		throw std::runtime_error("Error: Frame buffer isn't allocated yet.");
	}
}

}