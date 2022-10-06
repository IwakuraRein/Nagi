#ifndef SWAP_CHAIN_HPP
#define SWAP_CHAIN_HPP

#include "vk_device.hpp"

#define MAX_FRAMES_IN_FLIGHT 2

namespace nagi {

class SwapChain {
public:
	SwapChain(Device& deviceRef, VkExtent2D extent, VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR);
	SwapChain(Device& deviceRef, VkExtent2D extent, std::shared_ptr<SwapChain> pPrevious, VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR);
	~SwapChain();
	SwapChain(const SwapChain&) = delete;
	SwapChain& operator=(const SwapChain&) = delete;
	SwapChain() = default;

	VkDevice device() const { return _device.device(); };

	void createSwapChainColorAttachment(
		std::vector<VkImage>& images,
		std::vector<VkImageView>& views);

	size_t imageCount() { return _imageCount; }
	VkFormat getSwapChainImageFormat() { return _format; }
	VkExtent2D getSwapChainExtent() { return _swapChainExtent; }
	uint32_t width() { return _swapChainExtent.width; }
	uint32_t height() { return _swapChainExtent.height; }

	float extentAspectRatio() {
		return static_cast<float>(_swapChainExtent.width) / static_cast<float>(_swapChainExtent.height);
	}

	VkResult acquireNextImage(uint32_t* imageIndex);
	VkResult submitCommandBuffers(const VkCommandBuffer* buffers, uint32_t* imageIndex);

	friend class Renderer;
protected:
	void createSwapChain(VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR);
	void createSyncObjects();

	// Helper functions
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(
		const std::vector<VkSurfaceFormatKHR>& availableFormats,
		const VkFormat& format = VK_FORMAT_B8G8R8A8_UNORM,
		const VkColorSpaceKHR& colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR);
	VkPresentModeKHR chooseSwapPresentMode(
		const std::vector<VkPresentModeKHR>& availablePresentModes, VkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR);
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

	std::shared_ptr<SwapChain> _pOldSwapChain;
	VkExtent2D _swapChainExtent;

	Device& _device;
	VkExtent2D _windowExtent;

	uint32_t _imageCount;
	VkFormat _format;

	VkSwapchainKHR swapChain;

	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	std::vector<VkFence> imagesInFlight;
	size_t currentFrame = 0;
};

}  // namespace naku

#endif
