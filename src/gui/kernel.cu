#include "gui.hpp"

namespace nagi {

__global__ void kernCopyToFrameBuffer(uchar4* pbo, int width, int height, int pixels, float gamma, float* buffer) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    int py = idx / width;
    int px = idx - py * width;

    int idx2 = (height - py - 1) * width + px;
    //int idx2 = idx;

    pbo[idx2].x = glm::clamp((unsigned char)(powf(buffer[idx * 3], 1.f / gamma) * 255.f), (unsigned char)0, (unsigned char)255);
    pbo[idx2].y = glm::clamp((unsigned char)(powf(buffer[idx * 3 + 1], 1.f / gamma) * 255.f), (unsigned char)0, (unsigned char)255);
    pbo[idx2].z = glm::clamp((unsigned char)(powf(buffer[idx * 3 + 2], 1.f / gamma) * 255.f), (unsigned char)0, (unsigned char)255);
    pbo[idx2].w = 0;
}

void GUI::copyToFrameBuffer() {
    if (cudaGraphicsMapResources(1, &regesitered_pbo, NULL) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to map pbo.");
    }
    if (cudaGraphicsResourceGetMappedPointer((void**)&devFrameBuffer, NULL, regesitered_pbo) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to get device pointer.");
    }

    dim3 blocksPerGrid((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernCopyToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, width, height, pixels, gamma, devResultBuffer);

    cudaDeviceSynchronize();
    if (cudaGraphicsUnmapResources(1, &regesitered_pbo, NULL) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to unmap pbo.");
    }
}

}