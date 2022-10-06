#include "vk_swap_chain.hpp"

namespace nagi {

SwapChain::SwapChain(Device& deviceRef, VkExtent2D extent, std::shared_ptr<SwapChain> pPrevious, VkPresentModeKHR presentMode)
	: _device{ deviceRef }, _windowExtent{ extent }, _pOldSwapChain{ pPrevious } {
	createSwapChain(presentMode);
	createSyncObjects();
}

//TODO: modify color space
SwapChain::SwapChain(Device& deviceRef, VkExtent2D extent, VkPresentModeKHR presentMode)
	: _device{ deviceRef }, _windowExtent{ extent } {
	createSwapChain(presentMode);
	createSyncObjects();
}

SwapChain::~SwapChain() {

	if (swapChain != nullptr) {
		vkDestroySwapchainKHR(_device.device(), swapChain, nullptr);
		swapChain = nullptr;
	}

	// cleanup synchronization objects
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(_device.device(), renderFinishedSemaphores[i], nullptr);
		vkDestroySemaphore(_device.device(), imageAvailableSemaphores[i], nullptr);
		vkDestroyFence(_device.device(), inFlightFences[i], nullptr);
	}
}
void SwapChain::createSwapChainColorAttachment(
	std::vector<VkImage>& images,
	std::vector<VkImageView>& views) {
	images.resize(_imageCount);
	views.resize(_imageCount);
	vkGetSwapchainImagesKHR(_device.device(), swapChain, &_imageCount, images.data());
	//attachment.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;;
	for (int i = 0; i < imageCount(); i++) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = images[i];
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = _format;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;
		vkCreateImageView(_device.device(), &viewInfo, nullptr, &views[i]);
	}
}
VkResult SwapChain::acquireNextImage(uint32_t* imageIndex) {
	vkWaitForFences(
		_device.device(),
		1,
		&inFlightFences[currentFrame],
		VK_TRUE,
		std::numeric_limits<uint64_t>::max());

	VkResult result = vkAcquireNextImageKHR(
		_device.device(),
		swapChain,
		std::numeric_limits<uint64_t>::max(),
		imageAvailableSemaphores[currentFrame],  // must be a not signaled semaphore
		VK_NULL_HANDLE,
		imageIndex);

	return result;
}

VkResult SwapChain::submitCommandBuffers(
	const VkCommandBuffer* buffers, uint32_t* imageIndex) {
	if (imagesInFlight[*imageIndex] != VK_NULL_HANDLE) {
		vkWaitForFences(_device.device(), 1, &imagesInFlight[*imageIndex], VK_TRUE, UINT64_MAX);
	}
	imagesInFlight[*imageIndex] = inFlightFences[currentFrame];

	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
	VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submitInfo.waitSemaphoreCount = 1;
	submitInfo.pWaitSemaphores = waitSemaphores;
	submitInfo.pWaitDstStageMask = waitStages;

	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = buffers;

	VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
	submitInfo.signalSemaphoreCount = 1;
	submitInfo.pSignalSemaphores = signalSemaphores;

	vkResetFences(_device.device(), 1, &inFlightFences[currentFrame]);
	if (vkQueueSubmit(_device.graphicsQueue(), 1, &submitInfo, inFlightFences[currentFrame]) !=
		VK_SUCCESS) {
		throw std::runtime_error("failed to submit draw command buffer!");
	}

	VkPresentInfoKHR presentInfo = {};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = signalSemaphores;

	VkSwapchainKHR swapChains[] = { swapChain };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;

	presentInfo.pImageIndices = imageIndex;

	auto result = vkQueuePresentKHR(_device.presentQueue(), &presentInfo);

	currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

	return result;
}

void SwapChain::createSwapChain(VkPresentModeKHR mode) {
	SwapChainSupportDetails swapChainSupport = _device.getSwapChainSupport();

	VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
	VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes, mode);
	VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

	uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
	if (swapChainSupport.capabilities.maxImageCount > 0 and
		imageCount > swapChainSupport.capabilities.maxImageCount) {
		imageCount = swapChainSupport.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	createInfo.surface = _device.surface();

	createInfo.minImageCount = imageCount;
	createInfo.imageFormat = surfaceFormat.format;
	createInfo.imageColorSpace = surfaceFormat.colorSpace;
	createInfo.imageExtent = extent;
	createInfo.imageArrayLayers = 1;
	createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = _device.findPhysicalQueueFamilies();
	std::vector<uint32_t> queueFamilyIndices;
	queueFamilyIndices.push_back(indices.graphicsFamily);
	if (indices.graphicsFamily != indices.computeFamily) {
		queueFamilyIndices.push_back(indices.computeFamily);
	}
	if (indices.presentFamily != indices.graphicsFamily and indices.presentFamily != indices.computeFamily) {
		queueFamilyIndices.push_back(indices.presentFamily);
	}

	if (queueFamilyIndices.size() > 1) {
		createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
		createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
	}
	else {
		createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		createInfo.queueFamilyIndexCount = 0;      // Optional
		createInfo.pQueueFamilyIndices = nullptr;  // Optional
	}

	createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
	createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

	createInfo.presentMode = presentMode;
	createInfo.clipped = VK_TRUE;

	createInfo.oldSwapchain = _pOldSwapChain != nullptr ? _pOldSwapChain->swapChain : VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(_device.device(), &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	_format = surfaceFormat.format;
	_swapChainExtent = extent;
	// we only specified a minimum number of images in the swap chain, so the implementation is
	// allowed to create a swap chain with more. That's why we'll first query the final number of
	// images with vkGetSwapchainImagesKHR, then resize the container and finally call it again to
	// retrieve the handles.
	vkGetSwapchainImagesKHR(_device.device(), swapChain, &_imageCount, nullptr);
}

void SwapChain::createSyncObjects() {
	imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
	inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
	imagesInFlight.resize(imageCount(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphoreInfo = {};
	semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	VkFenceCreateInfo fenceInfo = {};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(_device.device(), &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) !=
			VK_SUCCESS or
			vkCreateSemaphore(_device.device(), &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) !=
			VK_SUCCESS or
			vkCreateFence(_device.device(), &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create synchronization objects for a frame!");
		}
	}
}

VkSurfaceFormatKHR SwapChain::chooseSwapSurfaceFormat(
	const std::vector<VkSurfaceFormatKHR>& availableFormats,
	const VkFormat& format,
	const VkColorSpaceKHR& colorSpace) {
	for (const auto& availableFormat : availableFormats) {
		if (availableFormat.format == format and
			availableFormat.colorSpace == colorSpace) {
			return availableFormat;
		}
	}

	return availableFormats[0];
}

VkPresentModeKHR SwapChain::chooseSwapPresentMode(
	const std::vector<VkPresentModeKHR>& availablePresentModes, VkPresentModeKHR mode) {

	for (const auto& availablePresentMode : availablePresentModes) {
		if (availablePresentMode == mode) {
			// std::cout << "Present mode: " << mode << std::endl;
			return availablePresentMode;
		}
	}

	std::cout << "warning: requested present mode not satisfied. Present mode: fifo" << std::endl;
	return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D SwapChain::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
	if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
		return capabilities.currentExtent;
	}
	else {
		VkExtent2D actualExtent = _windowExtent;
		actualExtent.width = std::max(
			capabilities.minImageExtent.width,
			std::min(capabilities.maxImageExtent.width, actualExtent.width));
		actualExtent.height = std::max(
			capabilities.minImageExtent.height,
			std::min(capabilities.maxImageExtent.height, actualExtent.height));

		return actualExtent;
	}
}

}
