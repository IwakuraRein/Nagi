#include "gui.hpp"

namespace nagi {

GUI::GUI(const std::string& windowName, PathTracer& pathTracer):
    pathTracer{ pathTracer }, scene{ pathTracer.scene }, wSize{ pathTracer.window }, gamma{ scene.config.gamma }, totalSpp{ scene.config.spp } {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        throw std::runtime_error("Error: Failed to initialize GLFW.");

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // Disable resizing window
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(wSize.width, wSize.height, windowName.c_str(), nullptr, nullptr);
    if (window == nullptr) {
        throw std::runtime_error("Error: Failed to create GLFW window.");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        throw std::runtime_error("Error: Failed to initialize GLAD");
    }

    glGenBuffers(1, &pbo); // make & register PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLubyte) * wSize.width * wSize.height, NULL, GL_DYNAMIC_DRAW);

    cudaRun(cudaGraphicsGLRegisterBuffer(&regesitered_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
}
void GUI::terminate() {
    if (window) {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaRun(cudaGraphicsUnregisterResource(regesitered_pbo));
        glDeleteBuffers(1, &pbo);

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();

        window = nullptr;
    }

    if (devLuminance) {
        cudaRun(cudaFree(devLuminance));
        devLuminance = nullptr;
    }
    if (devDenoisedResult1) {
        cudaRun(cudaFree(devDenoisedResult1));
        devDenoisedResult1 = nullptr;
    }
    if (devDenoisedResult2) {
        cudaRun(cudaFree(devDenoisedResult2));
        devDenoisedResult2 = nullptr;
    }
}

void GUI::render(float delta) {
	if (window) {
		if (glfwWindowShouldClose(window)) {
			terminate();
			return;
		}

        copyToFrameBuffer();

        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clearColor.x * clearColor.w, clearColor.y * clearColor.w, clearColor.z * clearColor.w, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);

        // display rendering result via OpenGL
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo); // THE MAGIC LINE #1 
        glDrawPixels(wSize.width, wSize.height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);   // THE MAGIC LINE #2

		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

        {
            ImGui::Begin("Preview Control Panel");

            ImGui::Text("Step          : %d.", step);
            static std::list<float> stack;
            stack.push_back(delta);
            if (stack.size() > 10) stack.pop_front();
            float avg = listSum(stack) / stack.size();

            ImGui::Text("Time remaining: %f sec.", avg * (totalSpp - step));

            ImGui::Text("Present"); ImGui::SameLine();
            static const char* items[] = { "Result", "Albedo", "Normal", "Depth"};
            ImGui::Combo("##Present", &present, items, 4);

            if (present != 2) {
                ImGui::Text("Gamma  "); ImGui::SameLine();
                ImGui::SliderFloat("##Gamma", &gamma, 0.0f, 5.0f);
            }
            if (present == 0) {
                ImGui::Text("Denoise"); ImGui::SameLine();
                ImGui::Checkbox("##Denoise", &denoiser);
                if (denoiser) {
                    ImGui::Text("Normal Weight   "); ImGui::SameLine();
                    ImGui::SliderFloat("##SigmaN", &sigmaN, 0.1f, 256.0f);
                    ImGui::Text("Depth Weight    "); ImGui::SameLine();
                    ImGui::SliderFloat("##SigmaZ", &sigmaZ, 0.1f, 10.0f);
                    ImGui::Text("Luminance Weight"); ImGui::SameLine();
                    ImGui::SliderFloat("##SigmaL", &sigmaL, 0.1f, 10.0f);
                }
            }

            ImGui::End();
        }

		ImGui::EndFrame();
		ImGui::Render();

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        step++;
    }
}


__global__ void kernCopyResultToFrameBuffer(uchar4* pbo, WindowSize window,float gamma, float* buffer, float blend) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= window.pixels) return;

    int py = idx / window.width;
    int px = idx - py * window.width;

    int idx2 = (window.height - py - 1) * window.width + px;
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
__global__ void kernCopyAlbedoToFrameBuffer(uchar4* pbo, WindowSize window,float gamma, float* albedo) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= window.pixels) return;

    int py = idx / window.width;
    int px = idx - py * window.width;

    int idx2 = (window.height - py - 1) * window.width + px;
    //int idx2 = idx;

    glm::vec3 color{ albedo[idx * 3], albedo[idx * 3 + 1], albedo[idx * 3 + 2] };
    color = glm::pow(color, glm::vec3{ 1.f / gamma }) * 255.f;

    pbo[idx2].x = glm::clamp((unsigned char)color.x, (unsigned char)0, (unsigned char)255);
    pbo[idx2].y = glm::clamp((unsigned char)color.y, (unsigned char)0, (unsigned char)255);
    pbo[idx2].z = glm::clamp((unsigned char)color.z, (unsigned char)0, (unsigned char)255);
    pbo[idx2].w = 0;
}
__global__ void kernCopyNormalToFrameBuffer(uchar4* pbo, WindowSize window,float* normal) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= window.pixels) return;

    int py = idx / window.width;
    int px = idx - py * window.width;

    int idx2 = (window.height - py - 1) * window.width + px;
    //int idx2 = idx;

    glm::vec3 color{ normal[idx * 3], normal[idx * 3 + 1], normal[idx * 3 + 2] };
    color = (color + 1.f) / 2.f;
    color *= 255.f;

    pbo[idx2].x = glm::clamp((unsigned char)color.x, (unsigned char)0, (unsigned char)255);
    pbo[idx2].y = glm::clamp((unsigned char)color.y, (unsigned char)0, (unsigned char)255);
    pbo[idx2].z = glm::clamp((unsigned char)color.z, (unsigned char)0, (unsigned char)255);
    pbo[idx2].w = 0;
}
__global__ void kernCopyDepthToFrameBuffer(uchar4* pbo, WindowSize window,float gamma, float* depth) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= window.pixels) return;

    int py = idx / window.width;
    int px = idx - py * window.width;

    int idx2 = (window.height - py - 1) * window.width + px;
    //int idx2 = idx;

    float z = depth[idx] * 255.f;
    z = glm::clamp(glm::pow(z, 1.f / gamma), 0.f, 255.f);

    pbo[idx2].x = (unsigned char)z;
    pbo[idx2].y = (unsigned char)z;
    pbo[idx2].z = (unsigned char)z;
    pbo[idx2].w = 0;
}

void GUI::copyToFrameBuffer() {
    cudaRun(cudaGraphicsMapResources(1, &regesitered_pbo, NULL));
    cudaRun(cudaGraphicsResourceGetMappedPointer((void**)&devFrameBuffer, NULL, regesitered_pbo))

    static bool isDenoised{ denoiser };
    dim3 blocksPerGrid((wSize.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (present == 0) { // result
        if (!denoiser) {
            isDenoised = false;
            kernCopyResultToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, wSize, gamma, pathTracer.devFrameBuf, 0.f);
        }
        else {
            denoise();
            float blend = isDenoised ? 0.5f : 0.f;
            kernCopyResultToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, wSize, gamma, devDenoisedResult2, blend);
            isDenoised = true;
        }
    }
    else if (present == 1) { // albedo
        isDenoised = false;
        kernCopyAlbedoToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, wSize, gamma, pathTracer.devAlbedoBuf);
    }
    else if (present == 2) { // normal
        isDenoised = false;
        kernCopyNormalToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, wSize, pathTracer.devNormalBuf);
    }
    else if (present == 3) { // depth
        isDenoised = false;
        kernCopyDepthToFrameBuffer <<<blocksPerGrid, BLOCK_SIZE>>>(devFrameBuffer, wSize, gamma, pathTracer.devDepthBuf);
    }

    cudaRun(cudaDeviceSynchronize());
    cudaRun(cudaGraphicsUnmapResources(1, &regesitered_pbo, NULL))
}

__global__ void kernGetY(int pixels, float* result, float* color) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= pixels) return;

    result[idx] = color[idx * 3] * 0.299f + color[idx * 3 + 1] * 0.587f + color[idx * 3 + 2] * 0.114f;
}
__global__ void kernDiscardColor(int pixels, float* result, float* color, float* albedo) {
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
    WindowSize window, int dilation, float* denoised, float* origin, float* luminance, float* luminance2, float* normal, float* depth, float sigmaN, float sigmaZ, float sigmaL) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= window.pixels) return;

    int py = idx / window.width;
    int px = idx - py * window.width;

    float variance{ 0.f };
    float var_w{ 0.f };
#pragma unroll
    for (int i = -1; i < 1; i++) {
#pragma unroll
        for (int j = -1; j < 1; j++) {
            int p = window.width * (py + i) + px + j;
            if (p >= 0 && p < window.pixels) {
                float y2 = luminance2[p];
                float y = luminance[p];
                variance += sqrtf(fabsf(y2 - y * y)) * devGaussianKernel3x3[4 + i * 3 + j];
                var_w += devGaussianKernel3x3[4 + i * 3 + j];
            }
        }
    }

    variance /= (1.f + variance) * var_w;
    variance = fmaxf(VAR_EPSILON, variance);

    //float thisL = luminance[idx];
    glm::vec3 thisL{ origin[idx * 3], origin[idx * 3+1], origin[idx * 3+2] };
    glm::vec3 thisN{ normal[idx * 3], normal[idx * 3 + 1], normal[idx * 3 + 2] };
    float thisZ{ depth[idx] };
    glm::vec3 out{ 0.f };
    glm::vec3 wSum{ 0.f };


#pragma unroll
    for (int i = -FILTER_SIZE_HALF; i <= FILTER_SIZE_HALF; i++) {
#pragma unroll
        for (int j = -FILTER_SIZE_HALF; j <= FILTER_SIZE_HALF; j++) {
            int p = window.width * (py + i * dilation) + px + j * dilation;
            if (p >= 0 && p < window.pixels) {
                glm::vec3 c{ origin[p * 3], origin[p * 3 + 1], origin[p * 3 + 2] };
                if (p == idx) {
                    out += c * devGaussianKernel[FILTER_AREA_HALF];
                    wSum += devGaussianKernel[FILTER_AREA_HALF];
                }
                else {
                    glm::vec3 n{ normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2] };
                    glm::vec3 l{ origin[p * 3], origin[p * 3 + 1], origin[p * 3 + 2] };
                    float z = depth[p];

                    float wn = powf(fmaxf(0.f, glm::dot(thisN, n)), sigmaN);
                    float wz_tmp = -fabsf(thisZ - z) / sigmaZ;
                    glm::vec3 diff{ thisL - l };
                    float sigmaL2 = sigmaL * variance + FLT_EPSILON;
                    glm::vec3 w{ __expf(-fabsf(diff.x) / sigmaL2 + wz_tmp),
                                 __expf(-fabsf(diff.y) / sigmaL2 + wz_tmp), 
                                 __expf(-fabsf(diff.z) / sigmaL2 + wz_tmp) };
                    //w = w * wn / (fabsf(i * dilation) + fabsf(j * dilation));
                    w = w * wn * devGaussianKernel[FILTER_AREA_HALF + i * FILTER_SIZE + j];
                    out += w * c;
                    wSum += w;
                }
            }
        }
    }
    out /= wSum;
    denoised[idx * 3] = out.x;
    denoised[idx * 3 + 1] = out.y;
    denoised[idx * 3 + 2] = out.z;
}

void GUI::denoise() {
    dim3 blocksPerGrid((wSize.pixels + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (!devLuminance) {
        cudaRun(cudaMalloc((void**)&devLuminance, sizeof(float) * wSize.pixels));
    }
    if (!devDenoisedResult1) {
        cudaRun(cudaMalloc((void**)&devDenoisedResult1, sizeof(float) * 3 * wSize.pixels));
    }
    if (!devDenoisedResult2) {
        cudaRun(cudaMalloc((void**)&devDenoisedResult2, sizeof(float) * 3 * wSize.pixels));
    }
    
    kernDiscardColor<<<blocksPerGrid, BLOCK_SIZE>>>(wSize.pixels, devDenoisedResult1, pathTracer.devFrameBuf, pathTracer.devAlbedoBuf);
    kernGetY<<<blocksPerGrid, BLOCK_SIZE>>>(wSize.pixels, devLuminance, devDenoisedResult1);
    
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(
		wSize, 1, devDenoisedResult2, devDenoisedResult1, devLuminance, pathTracer.devLumiance2Buf, pathTracer.devCurrentNormalBuf, pathTracer.devCurrentDepthBuf, sigmaN, sigmaZ, sigmaL);
	std::swap(devDenoisedResult1, devDenoisedResult2);
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(
		wSize, 2, devDenoisedResult2, devDenoisedResult1, devLuminance, pathTracer.devLumiance2Buf, pathTracer.devCurrentNormalBuf, pathTracer.devCurrentDepthBuf, sigmaN, sigmaZ, sigmaL);
	std::swap(devDenoisedResult1, devDenoisedResult2);
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(
		wSize, 4, devDenoisedResult2, devDenoisedResult1, devLuminance, pathTracer.devLumiance2Buf, pathTracer.devCurrentNormalBuf, pathTracer.devCurrentDepthBuf, sigmaN, sigmaZ, sigmaL);
	std::swap(devDenoisedResult1, devDenoisedResult2);
	kernBilateralFilter<<<blocksPerGrid, BLOCK_SIZE>>>(
		wSize, 8, devDenoisedResult2, devDenoisedResult1, devLuminance, pathTracer.devLumiance2Buf, pathTracer.devCurrentNormalBuf, pathTracer.devCurrentDepthBuf, sigmaN, sigmaZ, sigmaL);
	std::swap(devDenoisedResult1, devDenoisedResult2);

	kernRetrieveColor<<<blocksPerGrid, BLOCK_SIZE>>>(wSize.pixels, devDenoisedResult2, devDenoisedResult1, pathTracer.devAlbedoBuf);
}


}