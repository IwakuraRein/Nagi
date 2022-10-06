#include "gui.hpp"

using namespace ImGui;

namespace nagi {

GUI::GUI(int Width, int Height, const std::string& windowName)
	: width{ Width }, height{ Height }, pixels{ Width * Height }, windowExtend{ (uint32_t)Width, (uint32_t)Height }{
	glfwInit();

	// glfwWindowHint(int hint, int value): set value to hint
	// GLFW uses OpenGL as defualt. Disable this with GLFW_NO_API
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	// Disable resizing window
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	// The fourth parameter allows you to optionally specify a monitor to open the window on and the last parameter is only relevant to OpenGL.
	window = (glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr));
	glfwSetWindowUserPointer(window, this);

	std::cout << "Initializing GUI... " << std::endl;

	device = std::make_shared<Device>(window);

	{std::vector<VkDescriptorPoolSize> pool_sizes
		{
			{ VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
			{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
			{ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
	};

	descriptorPool = std::make_shared<DescriptorPool>(
		*device,
		1000,
		VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		pool_sizes
		);
	}

	{
		_currentImageIdx = 0;
		// wait until the current swapchain is not used
		vkDeviceWaitIdle(device->device());
		swapChain = std::make_shared<SwapChain>(*device, windowExtend, VK_PRESENT_MODE_FIFO_KHR);
	}

	{
		_commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = device->getCommandPool();
		allocInfo.commandBufferCount = static_cast<uint32_t>(_commandBuffers.size());

		if (vkAllocateCommandBuffers(device->device(), &allocInfo, _commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("Error: Failed to allocate command buffers.");
		}
	}

	{
		swapChain->createSwapChainColorAttachment(colorAttachment.images, colorAttachment.views);
		colorAttachment.format = swapChain->getSwapChainImageFormat();
		colorAttachment.clearValue.color = VkClearColorValue{ 0.f, 0.f, 0.f, 1.f };
		colorAttachment.extent = windowExtend;
		colorAttachment.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		RenderPassAttachment ColorAttachment{ colorAttachment };
		ColorAttachment.description.format = colorAttachment.format;
		ColorAttachment.description.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		renderPass = RenderPass::Builder(
			*device,
			*swapChain,
			{ ColorAttachment })
			.addSubPass({ 0 }, { }, -1)
			.build();
	}

	{
		CreateContext();
		ImGui_ImplGlfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo initInfo = {};
		initInfo.Instance = device->instance();
		initInfo.PhysicalDevice = device->physicalDevice();
		initInfo.Device = device->device();
		initInfo.Queue = device->graphicsQueue();
		initInfo.DescriptorPool = descriptorPool->getPool();
		initInfo.MinImageCount = device->getSwapChainSupport().capabilities.minImageCount;
		initInfo.ImageCount = swapChain->imageCount();
		initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
		initInfo.Subpass = 0;
		ImGui_ImplVulkan_Init(&initInfo, renderPass->renderPass());

		auto cmd = device->beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(cmd);
		device->endSingleTimeCommands(cmd);

		ImGui_ImplVulkan_DestroyFontUploadObjects();

		StyleColorsDark();
	}

	{
		frameImage.format = VK_FORMAT_R32G32B32_SFLOAT;

		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 1.f;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		//anisotropic_filtering
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 1.0f;

		if (vkCreateSampler(device->device(), &samplerInfo, nullptr, &frameImage.sampler) != VK_SUCCESS) {
			throw std::runtime_error("Error: Failed to create sampler!");
		}

		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1.f;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = frameImage.format;
		imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0;

		VmaAllocationCreateInfo allocCreateInfo{};
		allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

		if (vmaCreateImage(device->allocator(), &imageInfo, &allocCreateInfo, &frameImage.image, &frameImage.memory, nullptr)) {
			throw std::runtime_error("Error: failed to allocate image memory!");
		}

		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = frameImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = frameImage.format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device->device(), &viewInfo, nullptr, &frameImage.imageView) != VK_SUCCESS) {
			throw std::runtime_error("Error: failed to create image view!");
		}

		device->transitImageLayout(
			frameImage.image, frameImage.format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		frameImage.ImGuiImageId = 
			ImGui_ImplVulkan_AddTexture(frameImage.sampler, frameImage.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}

	std::cout << "GUI initilization finished." << std::endl;
}

GUI::~GUI()
{
	vkDeviceWaitIdle(device->device());

 	ImGui_ImplVulkan_Shutdown();
	vkFreeCommandBuffers(
		device->device(),
		device->getCommandPool(),
		static_cast<uint32_t>(_commandBuffers.size()),
		_commandBuffers.data());
	_commandBuffers.clear();
	for (int i = 0; i < colorAttachment.images.size(); i++) {
		vkDestroyImageView(device->device(), colorAttachment.views[i], nullptr);
	}
	colorAttachment.views.clear();
	vkDestroySampler(device->device(), frameImage.sampler, nullptr);
	vkDestroyImageView(device->device(), frameImage.imageView, nullptr);
	vmaDestroyImage(device->allocator(), frameImage.image, frameImage.memory);
}

void GUI::init() {
}

void GUI::render() {

	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	NewFrame();
	Image((ImTextureID)frameImage.ImGuiImageId, { (float)width, (float)height });
	ShowDemoWindow();
	EndFrame();
	Render();

	glfwPollEvents();
	auto commandBuffer = beginFrame();
	beginRenderPass(commandBuffer);
	ImGui_ImplVulkan_RenderDrawData(GetDrawData(), commandBuffer);
	//endRenderPass();
	vkCmdEndRenderPass(commandBuffer);

	endFrame();
}

VkCommandBuffer GUI::beginFrame() {
	assert(!_isFrameStarted and "Error: Frame has already started.");
	auto result = swapChain->acquireNextImage(&_currentImageIdx);

	// VK_ERROR_OUT_OF_DATE_KHR:
	// A surface has changed so that it is no longer compatible with the swapchain
	// and further presentation requests using the swapchain will fail. Engine must
	// query the new surface properties and recreate their swapchain.
	if (result == VK_ERROR_OUT_OF_DATE_KHR) { //recreate swap chain
		_currentImageIdx = 0;
		// wait until the current swapchain is not used
		vkDeviceWaitIdle(device->device());
		swapChain = std::make_shared<SwapChain>(*device, VkExtent2D{ (uint32_t)width, (uint32_t)height }, VK_PRESENT_MODE_FIFO_KHR);
	}

	if (result != VK_SUCCESS and result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Error: Failed to acquire swap chain image!");
	}

	_isFrameStarted = true;
	auto commandBuffer = _commandBuffers[_currentFrameIdx];
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
		throw std::runtime_error("Error: Failed to begin recording command buffer!");
	}
	return commandBuffer;

}

void GUI::endFrame() {
	assert(_isFrameStarted and "Error: Frame isn't in progress.");
	auto commandBuffer = _commandBuffers[_currentFrameIdx];
	if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
		throw std::runtime_error("failed to record command buffer!");
	}

	auto result = swapChain->submitCommandBuffers(&commandBuffer, &_currentImageIdx);
	if (result == VK_ERROR_OUT_OF_DATE_KHR ||
		result == VK_SUBOPTIMAL_KHR) {
		_currentImageIdx = 0;
		// wait until the current swapchain is not used
		vkDeviceWaitIdle(device->device());
		swapChain = std::make_shared<SwapChain>(*device, VkExtent2D{ (uint32_t)width, (uint32_t)height }, VK_PRESENT_MODE_FIFO_KHR);
	}


	_isFrameStarted = false;
	_currentFrameIdx = (_currentFrameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
}

void GUI::beginRenderPass(VkCommandBuffer commandBuffer)
{
	if (!_isFrameStarted) throw std::runtime_error("Error: Can't call begin render pass if frame is not in progress");
	if (!(commandBuffer == _commandBuffers[_currentFrameIdx])) throw std::runtime_error("Error: Can't begin render pass on command buffer from a different frame");

	glm::vec2 startCoord{ 0.f, 0.f };
	VkRenderPassBeginInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	renderPassInfo.renderPass = renderPass->renderPass();
	renderPassInfo.framebuffer = renderPass->frameBuffers[_currentImageIdx];

	//renderPassInfo.renderArea.offset = { static_cast<int>(startCoord.x), static_cast<int>(startCoord.y) };
	renderPassInfo.renderArea.offset = { 0, 0 };
	//renderPassInfo.renderArea.extent = _window.viewPortExtent();
	renderPassInfo.renderArea.extent = swapChain->getSwapChainExtent();

	renderPassInfo.clearValueCount = static_cast<uint32_t>(renderPass->clearValues.size());
	renderPassInfo.pClearValues = renderPass->clearValues.data();

	vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.x = 0.f;
	viewport.y = 0.f;
	viewport.width = (float)width;
	viewport.height = (float)height;

	viewport.minDepth = 0.f;
	viewport.maxDepth = 1.f;
	VkRect2D scissor{ {0,0}, windowExtend };
	vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
	vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

}