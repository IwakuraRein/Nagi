#include "gui.hpp"

namespace nagi {

__global__ void kernCopyToFrameBuffer(uchar4* pbo, int width, int height, int pixels, float gamma, float* buffer, float blend) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    int py = idx / width;
    int px = idx - py * width;

    int idx2 = (height - py - 1) * width + px;
    //int idx2 = idx;

    glm::vec3 color{ buffer[idx * 3], buffer[idx * 3 + 1], buffer[idx * 3 + 2] };
    color /= (1.f + color);
    color = glm::pow(color, glm::vec3{ 1.f / gamma }) * 255.f * (1.f - blend);

    color.x += pbo[idx2].x * blend;
    color.y += pbo[idx2].y * blend;
    color.z += pbo[idx2].z * blend;

    pbo[idx2].x = glm::clamp((unsigned char)color.x, (unsigned char)0, (unsigned char)255);
    pbo[idx2].y = glm::clamp((unsigned char)color.y, (unsigned char)0, (unsigned char)255);
    pbo[idx2].z = glm::clamp((unsigned char)color.z, (unsigned char)0, (unsigned char)255);
    pbo[idx2].w = 0;
}

void GUI::copyToFrameBuffer() {
    if (cudaGraphicsMapResources(1, &regesitered_pbo, NULL) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to map pbo.");
    }
    if (cudaGraphicsResourceGetMappedPointer((void**)&devFrameBuffer, NULL, regesitered_pbo) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to get device pointer.");
    }

    static bool isDenoised{ denoiser };
    dim3 blocksPerGrid((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (!denoiser) {
        isDenoised = false;
        kernCopyToFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, width, height, pixels, gamma, devResult, 0.f);
    }
    else {
        denoise();
        float blend = isDenoised ? 0.5f : 0.f;
        kernCopyToFrameBuffer<<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, width, height, pixels, gamma, devDenoisedResult2, blend);
        isDenoised = true;
    }

    cudaDeviceSynchronize();
    if (cudaGraphicsUnmapResources(1, &regesitered_pbo, NULL) != CUDA_SUCCESS) {
        throw std::runtime_error("Error: Failed to unmap pbo.");
    }
}

__global__ void kernGenerateLuminance(int pixels, float* result, float* color, float* albedo) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    float a1 = albedo[idx * 3] + FLT_EPSILON;
    float a2 = albedo[idx * 3 + 1] + FLT_EPSILON;
    float a3 = albedo[idx * 3 + 2] + FLT_EPSILON;

    result[idx * 3] = color[idx * 3] / a1;
    result[idx * 3 + 1] = color[idx * 3 + 1] / a2;
    result[idx * 3 + 2] = color[idx * 3 + 2] / a3;
}

__global__ void kernRetrieveColor(int pixels, float* result, float* luminance, float* albedo) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    float a1 = albedo[idx * 3] + FLT_EPSILON;
    float a2 = albedo[idx * 3 + 1] + FLT_EPSILON;
    float a3 = albedo[idx * 3 + 2] + FLT_EPSILON;

    result[idx * 3] = luminance[idx * 3] * a1;
    result[idx * 3 + 1] = luminance[idx * 3 + 1] * a2;
    result[idx * 3 + 2] = luminance[idx * 3 + 2] * a3;
}

// reference: https://dl.acm.org/doi/10.1145/3105762.3105770
//            http://diglib.eg.org/handle/10.2312/EGGH.HPG10.067-075

//todo: optimization with shared memeory
__global__ void kernBilateralFilter(
    int width, int height, int pixels, int dilation, float* denoised, float* luminance, float* normal, float* depth, float sigmaN, float sigmaZ, float sigmaL) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    int py = idx / width;
    int px = idx - py * width;

    glm::vec3 thisL{ luminance[idx * 3], luminance[idx * 3 + 1], luminance[idx * 3 + 2] };
    glm::vec3 thisN{ normal[idx * 3], normal[idx * 3 + 1], normal[idx * 3 + 2] };
    float thisZ{ depth[idx] };
    glm::vec3 out{ 0.f };
    glm::vec3 wSum{ 0.f };


#pragma unroll
    for (int i = -FILTER_SIZE_HALF; i <= FILTER_SIZE_HALF; i++) {
#pragma unroll
        for (int j = -FILTER_SIZE_HALF; j <= FILTER_SIZE_HALF; j++) {
            int p = width * (py + i * dilation) + px + j * dilation;
            if (p == idx) {
                out += thisL * devGaussianKernel[FILTER_AREA_HALF];
                wSum += devGaussianKernel[FILTER_AREA_HALF];
            }
            else if (p >= 0 && p < pixels) {
                glm::vec3 n{ normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2] };
                glm::vec3 l{ luminance[p * 3], luminance[p * 3 + 1], luminance[p * 3 + 2] };
                float z = depth[p];

                float wn = powf(fmaxf(0.f, glm::dot(thisN, n)), sigmaN);
                float wz_tmp = -fabsf(thisZ - z) / sigmaZ;
                glm::vec3 w{
                    __expf(-fabsf(thisL.x - l.x) / sigmaL + wz_tmp),
                    __expf(-fabsf(thisL.y - l.y) / sigmaL + wz_tmp),
                    __expf(-fabsf(thisL.z - l.z) / sigmaL + wz_tmp) };
                //w = w * wn / (fabsf(i * dilation) + fabsf(j * dilation));
                w = w * wn * devGaussianKernel[FILTER_AREA_HALF + i * FILTER_SIZE + j];
                out += w * l;
                wSum += w;
            }
        }
    }
    out /= wSum;
    denoised[idx * 3] = out.x;
    denoised[idx * 3 + 1] = out.y;
    denoised[idx * 3 + 2] = out.z;
}

void GUI::denoise() {
    dim3 blocksPerGrid((pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (!devDenoisedResult1) {
        cudaMalloc((void**)&devDenoisedResult1, sizeof(float) * 3 * pixels);
    }
    if (!devDenoisedResult2) {
        cudaMalloc((void**)&devDenoisedResult2, sizeof(float) * 3 * pixels);
    }
    
	kernGenerateLuminance<<<blocksPerGrid, BLOCK_SIZE>>>(pixels, devDenoisedResult1, devResult, devAlbedo);
    
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(width, height, pixels, 1, devDenoisedResult2, devDenoisedResult1, devNormal, devDepth, sigmaN, sigmaZ, sigmaL);
    std::swap(devDenoisedResult1, devDenoisedResult2);
    kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(width, height, pixels, 2, devDenoisedResult2, devDenoisedResult1, devNormal, devDepth, sigmaN, sigmaZ, sigmaL);
    std::swap(devDenoisedResult1, devDenoisedResult2);
    kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(width, height, pixels, 4, devDenoisedResult2, devDenoisedResult1, devNormal, devDepth, sigmaN, sigmaZ, sigmaL);
    std::swap(devDenoisedResult1, devDenoisedResult2);
    kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(width, height, pixels, 8, devDenoisedResult2, devDenoisedResult1, devNormal, devDepth, sigmaN, sigmaZ, sigmaL);
    std::swap(devDenoisedResult1, devDenoisedResult2);
    
    kernRetrieveColor<<<blocksPerGrid, BLOCK_SIZE>>>(pixels, devDenoisedResult2, devDenoisedResult1, devAlbedo);
}

}