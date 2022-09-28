#include "path_tracer.cuh"

#include <thrust/host_vector.h>  
#include <thrust/device_vector.h>
#include <thrust/scan.h>
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
	cudaMemset(devShadeFlags, 0, PIXEL_COUNT);

	dim3 fullBlocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeFrameBuffer<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devFrameBuf);
	checkCUDAError("kernInitializeFrameBuffer failed.");
	spp = 0;
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
	cudaMalloc((void**)&devResults1, PIXEL_COUNT * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults1 failed.");
	cudaMalloc((void**)&devResults2, PIXEL_COUNT * sizeof(IntersectInfo));
	checkCUDAError("cudaMalloc devResults2 failed.");
	cudaMalloc((void**)&devShadeFlags, PIXEL_COUNT * sizeof(int));
	checkCUDAError("cudaMalloc devInitialRayPool failed.");
}

void PathTracer::destroyBuffers() {
	if (devObjBuf) {
		cudaFree(devObjBuf);
		checkCUDAError2("cudaFree devObjBuf failed.");
	}
	if (devMtlBuf) {
		cudaFree(devMtlBuf);
		checkCUDAError2("cudaFree devMtlBuf failed.");
	}
	if (devTrigBuf) {
		cudaFree(devTrigBuf);
		checkCUDAError2("cudaFree devTrigBuf failed.");
	}
	if (devRayPool1) {
		cudaFree(devRayPool1);
		checkCUDAError2("cudaFree devRayPool1 failed.");
	}
	if (devRayPool2) {
		cudaFree(devRayPool2);
		checkCUDAError2("cudaFree devRayPool2 failed.");
	}
	if (devResults1) {
		cudaFree(devResults1);
		checkCUDAError2("cudaFree devResults1 failed.");
	}
	if (devResults2) {
		cudaFree(devResults2);
		checkCUDAError2("cudaFree devResults2 failed.");
	}
	if (devShadeFlags) {
		cudaFree(devShadeFlags);
		checkCUDAError2("cudaFree devShadeFlags failed.");
	}
	if (devFrameBuf) {
		cudaFree(devFrameBuf);
		checkCUDAError2("cudaFree devFrameBuf failed.");
	}
}

PathTracer::~PathTracer() {
	destroyBuffers();
}

// intersection test -> compact rays -> sort rays according to material -> compute color -> compact rays -> intersection test...
void PathTracer::iterate() {
	for (; spp < scene.config.spp; spp++) {
		dim3 fullBlocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
		kernInitializeRays<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devRayPool1, scene.config.maxBounce, scene.cam);
		checkCUDAError("kernInitializeRays failed.");
		remainingRays = PIXEL_COUNT;
		while (remainingRays > 0) {
			intersectionTest(devRayPool1, devResults1);
			remainingRays = compactRays(devRayPool1, devRayPool2, devResults1, devResults2);
			std::swap(devRayPool1, devRayPool2);
			std::swap(devResults1, devResults2);
			sort(devRayPool1, devRayPool2, devResults1, devResults2);
			std::swap(devRayPool1, devRayPool2);
			std::swap(devResults1, devResults2);
			shade(devResults1, devShadeFlags);
			remainingRays = compactRays(devRayPool1, devRayPool2, devShadeFlags);
		}
	}
}

void PathTracer::intersectionTest(Path* rayPool, IntersectInfo* results) {
	
}

int PathTracer::compactRays(Path* rayPool, Path* compactedRayPool, IntersectInfo* intersectResults, IntersectInfo* compactedIntersectResults) {
	// delete rays that hit nothing. leave zero in the frame buffer
	// thrust here
	return 0;
}
int PathTracer::compactRays(Path* rayPool, Path* compactedRayPool, int* flags) {
	// delete rays that hit light source. write color into the frame buffer
	// thrust here
	return 0;
}

void PathTracer::sort(Path* rayPool, Path* sortedRayPool, IntersectInfo* intersectResults, IntersectInfo* sortedIntersectResults) {
	// sort rays according to materials
	// thrust here
}

void PathTracer::shade(IntersectInfo* intersectResults, int* flags) {
	// compute color and generate new ray direction
	// if ray should terminate, write -1 into the flag array
}

//__global__ void kernInitializeRays(Path* rayPool, int maxBounce, const glm::vec3 camPos, const glm::mat4 invProjViewMat) {
__global__ void kernInitializeRays(Path* rayPool, int maxBounce, const Camera cam) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;
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
	path.ray.dir = cam.screenOrigin - cam.upDir * (py * cam.pixelHeight + cam.pixelHeight / 2) + cam.rightDir * (px * cam.pixelWidth + cam.pixelWidth / 2);
	path.ray.dir = glm::normalize(path.ray.dir);
	path.ray.invDir = 1.f / path.ray.dir;
	rayPool[idx] = path;
}

__global__ void kernInitializeFrameBuffer(float* frame) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;
	frame[idx * 3] = 0.f;
	frame[idx * 3 + 1] = 0.f;
	frame[idx * 3 + 2] = 0.f;
}

__global__ void kernTest(Path* rayPool, Triangle* trigBuf, int num, float* frameBuffer) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (idx >= PIXEL_COUNT) return;
	Ray r = rayPool[idx].ray;
	glm::vec3 normal;
	glm::vec3 pickedNormal{ 0.f, 0.f, 0.f };
	float dist;
	float minDist{ FLT_MAX };
	bool found{ false };
	for (int i = 0; i < num; i++) {
		Triangle trig = trigBuf[i];
		if (rayBoxIntersect(r, trig.bbox, &dist)) {
			if (rayTrigIntersect(r, trig, &dist, &normal)) {
				if (dist < minDist) {
					minDist = dist;
					pickedNormal = normal;
					found = true;
				}
			}
		}
	}
	if (found) {
		frameBuffer[idx * 3] = (pickedNormal.x + 1.f) / 2.f;
		frameBuffer[idx * 3 + 1] = (pickedNormal.y + 1.f) / 2.f;
		frameBuffer[idx * 3 + 2] = (pickedNormal.z + 1.f) / 2.f;
	}
}

void PathTracer::test1(float* frameBuffer) {
	dim3 fullBlocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeRays << <fullBlocksPerGrid, BLOCK_SIZE >> > (devRayPool1, scene.config.maxBounce, scene.cam);
	checkCUDAError("kernInitializeRays failed.");
	kernTest<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devRayPool1, devTrigBuf, scene.trigBuf.size(), devFrameBuf);
	checkCUDAError("kernTest failed.");
	cudaMemcpy(frameBuffer, devFrameBuf, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devFrameBuffer failed.");
}

}