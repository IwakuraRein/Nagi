#ifndef GUI_HPP
#define GUI_HPP

#include "vk_device.hpp"
#include "vk_descriptor.hpp"
#include "vk_swap_chain.hpp"
#include "vk_render_pass.hpp"

#define IM_VEC2_CLASS_EXTRA constexpr ImVec2(const glm::vec2& f) : x(f.x), y(f.y) {} operator glm::vec2() const { return glm::vec2(x,y); }
#define IM_VEC4_CLASS_EXTRA constexpr ImVec4(const glm::vec4& f) : x(f.x), y(f.y), z(f.z), w(f.w) {} operator glm::vec4() const { return glm::vec4(x,y,z,w); }
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <memory>
#include <string>

namespace nagi {

class GUI {
public:
	GUI(int Width, int Height, const std::string& windowName);
	~GUI();
	GUI(const GUI&) = delete;
	void operator=(const GUI&) = delete;

	void init();
	void render();
	void print(const std::string& info) {}
	VkCommandBuffer beginFrame();
	void endFrame();
	void beginRenderPass(VkCommandBuffer commandBuffer);

	int width, height, pixels;
	VkExtent2D windowExtend;
	GLFWwindow* window = nullptr;

	bool _isFrameStarted{ false };
	uint32_t _currentImageIdx{ 0 };
	uint32_t _currentFrameIdx{ 0 };
	std::shared_ptr<Device> device;
	std::shared_ptr<DescriptorPool> descriptorPool;
	std::shared_ptr<DescriptorSetLayout> descriptorSetLayout;
	std::shared_ptr<SwapChain> swapChain;
	std::shared_ptr<RenderPass> renderPass;
	std::vector<VkCommandBuffer> _commandBuffers;

	FrameBufferAttachment colorAttachment;

	struct {
		VkFormat format;
		VmaAllocation memory;
		VkImageView imageView;
		VkImage image;
		VkSampler sampler;
		VkDescriptorSet ImGuiImageId;
	} frameImage;
};

}

#endif // !GUI_HPP
