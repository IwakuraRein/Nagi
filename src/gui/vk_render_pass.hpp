#ifndef RENDER_PASS_HPP
#define RENDER_PASS_HPP

#include "vk_device.hpp"
#include "vk_swap_chain.hpp"
#include "vk_descriptor.hpp"

namespace nagi {

struct FrameBufferAttachment {
	VkFormat format;
	VkExtent2D extent;
	VkImageUsageFlags usage;
	VkClearValue clearValue{};
	std::vector<VkImage> images;
	//std::vector<VkDeviceMemory> mems;
	std::vector<VmaAllocation> allocs;
	std::vector<VkImageView> views;

	void clear(Device& device)
	{
		for (int i = 0; i < images.size(); i++) {
			vkDestroyImageView(device.device(), views[i], nullptr);
			vmaDestroyImage(device.allocator(), images[i], allocs[i]);
		}
	}
};

struct RenderPassAttachment {
	FrameBufferAttachment& attachment;
	VkAttachmentDescription description{
		0,
		VK_FORMAT_UNDEFINED,
		VK_SAMPLE_COUNT_1_BIT,
		VK_ATTACHMENT_LOAD_OP_CLEAR,
		VK_ATTACHMENT_STORE_OP_STORE,
		VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		VK_ATTACHMENT_STORE_OP_STORE,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
	};
};

class RenderPass {
public:
	struct Subpass {
		uint32_t subpass;
		VkSubpassDescription subpassDescription;
		std::vector<FrameBufferAttachment> colorAttachments;
		std::vector<FrameBufferAttachment> inputAttachments;
		FrameBufferAttachment depthAttachment;

		// for setting up descriptor sets
		std::vector<VkImageLayout> inputImageLayouts;

		struct CreateInfo {
			std::vector<VkAttachmentReference> colorRef;
			std::vector<VkAttachmentReference> inputRef;
			bool hasDepth;
			VkAttachmentReference depthRef;
		} createInfo;

		VkSubpassDescription& build();
	};
	class Builder {
	public:
		Builder(
			Device& device,
			SwapChain& swapChain,
			const std::vector<RenderPassAttachment>& attachments);
		Builder& addSubPass(
			const std::vector<uint32_t>& colorAttachments,
			const std::vector<uint32_t>& inputAttachments,
			const int depthAttachment,
			bool depthTestOnly = false);
		Builder& addSubPass(
			const std::vector<uint32_t>& colorAttachments,
			const std::vector<uint32_t>& inputAttachments,
			const int depthAttachment,
			const std::vector<VkImageLayout>& colorAttachmentLayouts,
			const std::vector<VkImageLayout>& inputAttachmentLayouts,
			const VkImageLayout depthAttachmentLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
		std::unique_ptr<RenderPass> build();
	private:
		uint32_t _subpassCount{ 0 };
		Device& _device;
		SwapChain& _swapChain;
		std::vector<FrameBufferAttachment> _attachments;
		std::vector<VkAttachmentDescription> _attachmentDescriptions;
		std::vector<Subpass> _subpasses;
		std::vector<VkClearValue> _clearValues;
		std::vector<VkSubpassDependency> _dependencies;
		//std::vector<VkPipelineColorBlendAttachmentState> _blendAttachments;
		void setupDependencies();
		//void setupSubPassReferences();
		void createFramebuffers(
			std::vector<VkFramebuffer>& frameBuffers,
			VkRenderPass renderPass,
			const std::vector<FrameBufferAttachment>& attachMents);
	};
	RenderPass(
		Device& device,
		SwapChain& swapChain)
		: _device{ device }, _swapChain{ swapChain } {}

	~RenderPass() {
		vkDestroyRenderPass(_device.device(), _pass, nullptr);
		for (auto framebuffer : frameBuffers) {
			vkDestroyFramebuffer(_device.device(), framebuffer, nullptr);
		}
	}
	RenderPass(const RenderPass&&) = delete;
	void operator=(const RenderPass&&) = delete;

	VkRenderPass renderPass() const { return _pass; }
	size_t imageCount() const { return attachments[0].images.size(); }

	std::vector<VkFramebuffer> frameBuffers;
	std::vector<Subpass> subpasses;
	std::vector<FrameBufferAttachment> attachments;
	std::vector<VkClearValue> clearValues;
	std::vector<VkPipelineColorBlendAttachmentState> blendAttachments;
private:
	Device& _device;
	SwapChain& _swapChain;
	VkRenderPass _pass;
};

}

#endif