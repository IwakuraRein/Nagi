#include "path_tracer.cuh"

namespace nagi {

void PathTracer::initialize() {
	cudaMalloc((void**)&devObjBuf, scene.objBuf.size() * sizeof(Object));
	checkCUDAError("cudaMalloc devObjBuf failed.");
	cudaMalloc((void**)&devMtlBuf, scene.mtlBuf.size() * sizeof(Material));
	checkCUDAError("cudaMalloc devMtlBuf failed.");
	cudaMalloc((void**)&devTrigBuf, scene.trigBuf.size() * sizeof(Triangle));
	checkCUDAError("cudaMalloc devTrigBuf failed.");
	cudaMalloc((void**)&devFrameBuf, sizeof(float) * PIXEL_COUNT * 3);
	checkCUDAError("cudaMalloc devFrameBuf failed.");
	cudaMalloc((void**)&devRayPool, PIXEL_COUNT * sizeof(Path));
	checkCUDAError("cudaMalloc devRayPool failed.");

	cudaMemcpy(devObjBuf, scene.objBuf.data(), scene.objBuf.size() * sizeof(Object), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devObjBuf failed.");
	cudaMemcpy(devMtlBuf, scene.mtlBuf.data(), scene.mtlBuf.size() * sizeof(Material), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devMtlBuf failed.");
	cudaMemcpy(devTrigBuf, scene.trigBuf.data(), scene.trigBuf.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devTrigBuf failed.");

	dim3 fullBlocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernInitializeFrameBuffer<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devFrameBuf);
	checkCUDAError("kernInitializeFrameBuffer failed.");
	kernInitializeRays<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devRayPool, scene.config.maxBounce, scene.cam);
	checkCUDAError("kernInitializeRays failed.");
}

PathTracer::~PathTracer() {
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
	if (devRayPool) {
		cudaFree(devRayPool);
		checkCUDAError2("cudaFree devRayPool failed.");
	}
	if (devFrameBuf) {
		cudaFree(devFrameBuf);
		checkCUDAError2("cudaFree devFrameBuf failed.");
	}
}

void PathTracer::compactRays() {

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
	path.ray.dir = glm::normalize(path.ray.dir + FLT_EPSILON); // we don't allow dir to be parallel to the axises
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
	float dist;
	float minDist{ FLT_MAX };
	Triangle picked;
	glm::vec3 pickedNormal{ 0.f };
	for (int i = 0; i < num; i++) {
		Triangle trig = trigBuf[i];
		//if (rayBoxIntersect(r, trig.bbox, &dist)) {
		//	frameBuffer[idx * 3] = 1.f;
		//	frameBuffer[idx * 3 + 1] = 1.f;
		//	frameBuffer[idx * 3 + 2] = 1.f;
		//	return;
		//}
		if (rayTrigIntersect(r, trig, &dist, &normal)) {
			if (dist < minDist) {
				minDist = dist;
				picked = trig;
				pickedNormal = normal;
			}
		}
	}
	frameBuffer[idx * 3] = (pickedNormal.x + 1.f) / 2.f;
	frameBuffer[idx * 3 + 1] = (pickedNormal.y + 1.f) / 2.f;
	frameBuffer[idx * 3 + 2] = (pickedNormal.z + 1.f) / 2.f;
}

void PathTracer::test1(float* frameBuffer) {
	dim3 fullBlocksPerGrid((PIXEL_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE);
	kernTest<<<fullBlocksPerGrid, BLOCK_SIZE>>>(devRayPool, devTrigBuf, scene.trigBuf.size(), devFrameBuf);
	checkCUDAError("kernTest failed.");
	cudaMemcpy(frameBuffer, devFrameBuf, sizeof(float) * PIXEL_COUNT * 3, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy devFrameBuffer failed.");
}

}