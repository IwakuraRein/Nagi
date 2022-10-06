#include "vk_render_pass.hpp"

#include <set>

namespace nagi {

void RenderPass::Builder::createFramebuffers(
	std::vector<VkFramebuffer>& frameBuffers,
	VkRenderPass renderPass,
	const std::vector<FrameBufferAttachment>& attachMents)
{
	if (frameBuffers.size() > 0) {
		throw std::runtime_error("Error: Frame Buffers are not empty. Destroy all frame buffers before creation.");
	}
	frameBuffers.resize(_swapChain.imageCount());
	uint32_t width{ std::numeric_limits<uint32_t>::max() }, height{ std::numeric_limits<uint32_t>::max() };
	for (auto att : attachMents) {
		if (att.extent.width < width) width = att.extent.width;
		if (att.extent.height < height) height = att.extent.height;
	}
	for (int i = 0; i < _swapChain.imageCount(); i++) {
		std::vector<VkImageView> imageViews;
		imageViews.reserve(attachMents.size());
		for (auto att : attachMents) {
			if (att.views.size() == _swapChain.imageCount())
				imageViews.push_back(att.views[i]);
			else imageViews.push_back(att.views[0]);
		}
		//VkExtent2D swapChainExtent = _swapChain.getSwapChainExtent();
		VkFramebufferCreateInfo framebufferInfo = {};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = renderPass;
		framebufferInfo.attachmentCount = static_cast<uint32_t>(attachMents.size());
		framebufferInfo.pAttachments = imageViews.data();
		//framebufferInfo.width = swapChainExtent.width;
		//framebufferInfo.height = swapChainExtent.height;
		framebufferInfo.width = width;
		framebufferInfo.height = height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(
			_device.device(),
			&framebufferInfo,
			nullptr,
			&frameBuffers[i]) != VK_SUCCESS) {
			throw std::runtime_error("Error: Failed to create framebuffer!");
		}
	}
}

RenderPass::Builder::Builder(
	Device& device,
	SwapChain& swapChain,
	const std::vector<RenderPassAttachment>& attachments)
	: _device{ device }, _swapChain{ swapChain } {

	_attachments.reserve(attachments.size());
	_attachmentDescriptions.reserve(attachments.size());

	for (size_t i = 0; i < attachments.size(); i++) {
		_attachments.push_back(attachments[i].attachment);
		_attachmentDescriptions.push_back(attachments[i].description);
		_clearValues.push_back(attachments[i].attachment.clearValue);
	}
}
VkSubpassDescription& RenderPass::Subpass::build()
{
	subpassDescription = VkSubpassDescription{};
	subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpassDescription.colorAttachmentCount = static_cast<uint32_t>(createInfo.colorRef.size());
	subpassDescription.pColorAttachments = createInfo.colorRef.data();
	subpassDescription.inputAttachmentCount = static_cast<uint32_t>(createInfo.inputRef.size());
	subpassDescription.pInputAttachments = createInfo.inputRef.data();
	if (createInfo.hasDepth)
		subpassDescription.pDepthStencilAttachment = &createInfo.depthRef;
	else
		subpassDescription.pDepthStencilAttachment = nullptr;

	return subpassDescription;
}
RenderPass::Builder& RenderPass::Builder::addSubPass(
	const std::vector<uint32_t>& colorAttachments,
	const std::vector<uint32_t>& inputAttachments,
	const int depth,
	bool depthTestOnly)
{
	// attachments that will be both read && written
	std::set<uint32_t> generalAttachments;
	for (uint32_t i : colorAttachments) {
		if (find(inputAttachments.begin(), inputAttachments.end(), i) != inputAttachments.end())
			generalAttachments.insert(i);
	}
	uint32_t depthAttachment;
	if (depth >= 0) depthAttachment = static_cast<uint32_t>(depth);
	if (depth >= 0 && find(inputAttachments.begin(), inputAttachments.end(), depthAttachment) != inputAttachments.end()) {
		generalAttachments.insert(depthAttachment);
	}


	Subpass subpass{};
	subpass.subpass = _subpassCount++;

	subpass.createInfo.colorRef.reserve(colorAttachments.size());
	subpass.colorAttachments.reserve(colorAttachments.size());
	for (uint32_t i : colorAttachments) {
		if (generalAttachments.count(i) == 0)
			subpass.createInfo.colorRef.push_back({ i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		else
			subpass.createInfo.colorRef.push_back({ i, VK_IMAGE_LAYOUT_GENERAL });
		subpass.colorAttachments.push_back(_attachments[i]);
	}
	subpass.createInfo.inputRef.reserve(inputAttachments.size());
	subpass.inputAttachments.reserve(inputAttachments.size());
	subpass.inputImageLayouts.reserve(inputAttachments.size());
	for (uint32_t i : inputAttachments) {
		if (generalAttachments.count(i) == 0) {
			subpass.createInfo.inputRef.push_back({ i, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
			subpass.inputImageLayouts.push_back(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		else {
			subpass.createInfo.inputRef.push_back({ i, VK_IMAGE_LAYOUT_GENERAL });
			subpass.inputImageLayouts.push_back(VK_IMAGE_LAYOUT_GENERAL);
		}
		subpass.inputAttachments.push_back(_attachments[i]);
	}

	subpass.createInfo.hasDepth = depth >= 0;

	if (depth >= 0 && find(generalAttachments.begin(), generalAttachments.end(), depthAttachment) == generalAttachments.end()) {
		subpass.createInfo.depthRef = VkAttachmentReference{};
		subpass.createInfo.depthRef.attachment = depthAttachment;
		if (depthTestOnly)
			subpass.createInfo.depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
		else
			subpass.createInfo.depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		subpass.depthAttachment = _attachments[depthAttachment];
	}
	if (depth >= 0 && find(generalAttachments.begin(), generalAttachments.end(), depthAttachment) != generalAttachments.end()) {
		subpass.createInfo.depthRef.layout = VK_IMAGE_LAYOUT_GENERAL;
		subpass.depthAttachment = _attachments[depthAttachment];
	}

	_subpasses.push_back(std::move(subpass));

	return *this;
}
RenderPass::Builder& RenderPass::Builder::addSubPass(
	const std::vector<uint32_t>& colorAttachments,
	const std::vector<uint32_t>& inputAttachments,
	const int depth,
	const std::vector<VkImageLayout>& colorAttachmentLayouts,
	const std::vector<VkImageLayout>& inputAttachmentLayouts,
	const VkImageLayout depthAttachmentLayout)
{
	assert(colorAttachments.size() == colorAttachmentLayouts.size());
	assert(inputAttachments.size() == inputAttachmentLayouts.size());

	uint32_t depthAttachment;
	if (depth >= 0) depthAttachment = static_cast<uint32_t>(depth);

	Subpass subpass{};
	subpass.subpass = _subpassCount++;

	subpass.createInfo.colorRef.reserve(colorAttachments.size());
	subpass.colorAttachments.reserve(colorAttachments.size());
	for (uint32_t i : colorAttachments) {
		subpass.createInfo.colorRef.push_back({ i, colorAttachmentLayouts[i] });
		subpass.colorAttachments.push_back(_attachments[i]);
	}
	subpass.createInfo.inputRef.reserve(inputAttachments.size());
	subpass.inputAttachments.reserve(inputAttachments.size());
	subpass.inputImageLayouts.reserve(inputAttachments.size());
	for (uint32_t i : inputAttachments) {
		subpass.createInfo.inputRef.push_back({ i, inputAttachmentLayouts[i] });
		subpass.inputImageLayouts.push_back(inputAttachmentLayouts[i]);
		subpass.inputAttachments.push_back(_attachments[i]);
	}

	subpass.createInfo.hasDepth = depth >= 0;

	if (depth >= 0) {
		subpass.createInfo.depthRef = VkAttachmentReference{};
		subpass.createInfo.depthRef.attachment = depthAttachment;
		subpass.createInfo.depthRef.layout = depthAttachmentLayout;
		subpass.depthAttachment = _attachments[depthAttachment];
	}

	_subpasses.push_back(std::move(subpass));

	return *this;
}
void RenderPass::Builder::setupDependencies() {
	if (_subpasses.size() == 1) { //only 1 subpass
		_dependencies.resize(1);
		_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		_dependencies[0].srcAccessMask = 0;
		_dependencies[0].srcStageMask =
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		_dependencies[0].dstSubpass = 0;
		_dependencies[0].dstStageMask =
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		_dependencies[0].dstAccessMask =
			VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	}
	else {
		_dependencies.resize(_subpasses.size() + 1);
		VkSubpassDependency& firstDependency = _dependencies[0];
		firstDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		firstDependency.dstSubpass = 0;
		firstDependency.srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		firstDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		firstDependency.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		firstDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		firstDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		for (size_t i = 1; i < (_dependencies.size() - 1); i++) {
			_dependencies[i].srcSubpass = i - 1;
			_dependencies[i].dstSubpass = i;
			_dependencies[i].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			_dependencies[i].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
			_dependencies[i].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
			_dependencies[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			_dependencies[i].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
		}
		VkSubpassDependency& lastDependency = *(_dependencies.end() - 1);
		lastDependency.srcSubpass = _subpasses.size() - 1;
		lastDependency.dstSubpass = VK_SUBPASS_EXTERNAL;
		lastDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		lastDependency.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		lastDependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		lastDependency.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
		lastDependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
	}
}

std::unique_ptr<RenderPass> RenderPass::Builder::build() {
	auto renderpass = std::make_unique<RenderPass>(_device, _swapChain);

	setupDependencies();

	std::vector<VkSubpassDescription> subpassDescriptions;
	subpassDescriptions.reserve((_subpasses.size()));
	for (auto& s : _subpasses) {
		subpassDescriptions.push_back(s.build());
	}

	VkRenderPassCreateInfo renderPassInfo = {};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	renderPassInfo.attachmentCount = static_cast<uint32_t>(_attachmentDescriptions.size());
	renderPassInfo.pAttachments = _attachmentDescriptions.data();
	renderPassInfo.subpassCount = static_cast<uint32_t>(subpassDescriptions.size());
	renderPassInfo.pSubpasses = subpassDescriptions.data();
	renderPassInfo.dependencyCount = static_cast<uint32_t>(_dependencies.size());
	renderPassInfo.pDependencies = _dependencies.data();
	if (vkCreateRenderPass(_device.device(), &renderPassInfo, nullptr, &renderpass->_pass) != VK_SUCCESS) {
		throw std::runtime_error("Error: Failed to create render pass!");
	}

	createFramebuffers(renderpass->frameBuffers, renderpass->_pass, _attachments);

	renderpass->subpasses = std::move(_subpasses);
	renderpass->clearValues = std::move(_clearValues);
	//renderpass->blendAttachments = std::move(_blendAttachments);
	renderpass->attachments = std::move(_attachments);
	return renderpass;
}

}

