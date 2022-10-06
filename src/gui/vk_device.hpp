#ifndef DEVICE_HPP
#define DEVICE_HPP

#define GLM_COORDINATE_SYSTEM GLM_RIGHT_HANDED
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vma/vk_mem_alloc.h>

#include <iostream>
#include <string>
#include <vector>
#include <exception>

namespace nagi {

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
	uint32_t graphicsFamily;
	uint32_t presentFamily;
	uint32_t computeFamily;
	bool hasGraphicsFamily = false;
	bool hasPresentFamily = false;
	bool hasComputeFamily = false;
	bool isComplete() { return hasGraphicsFamily && hasPresentFamily && hasComputeFamily; }
};

class Device {
public:

#ifdef NDEBUG
	const bool enableValidationLayers = false;
#else
	const bool enableValidationLayers = true;
#endif
	Device(GLFWwindow* window);
	~Device();

	Device(const Device&) = delete;
	Device& operator=(const Device&) = delete;
	Device(Device&&) = delete;
	Device& operator=(Device&&) = delete;
	Device() = default;

	VkCommandPool getCommandPool() { return _commandPool; }
	VkDevice device() { return _device; }
	VmaAllocator allocator() { return _allocator; }
	VkSurfaceKHR surface() { return _surface; }
	VkQueue graphicsQueue() { return _graphicsQueue; }
	VkQueue presentQueue() { return _presentQueue; }
	VkQueue computeQueue() { return _computeQueue; }
	VkInstance instance() { return _instance; }
	VkPhysicalDevice physicalDevice() { return _physicalDevice; }

	SwapChainSupportDetails getSwapChainSupport() { return querySwapChainSupport(_physicalDevice); }
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	QueueFamilyIndices findPhysicalQueueFamilies() { return findQueueFamilies(_physicalDevice); }
	VkFormat findSupportedFormat(
		const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	// Buffer Helper Functions

	VkCommandBuffer beginSingleTimeCommands();
	void endSingleTimeCommands(VkCommandBuffer commandBuffer);
	void transitImageLayout(VkImage image,
		VkFormat format,
		VkImageLayout oldLayout,
		VkImageLayout newLayout,
		uint32_t baseMipLevel = 0,
		uint32_t levelCount = 1,
		uint32_t baseArrayLayer = 0,
		uint32_t layerCount = 1);

	VkPhysicalDeviceProperties properties;

private:
	void createInstance();
	void setupDebugMessenger();
	void createSurface();
	void pickPhysicalDevice();
	void createLogicalDevice();
	void createCommandPool();
	void createAllocator();

	// helper functions
	bool isDeviceSuitable(VkPhysicalDevice device);
	std::vector<const char*> getRequiredExtensions();
	bool checkValidationLayerSupport();
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
	void hasGflwRequiredInstanceExtensions();
	bool checkDeviceExtensionSupport(VkPhysicalDevice device);
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debugMessenger;
	VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
	VkCommandPool _commandPool;

	VkDevice _device;
	GLFWwindow* _window;
	VkSurfaceKHR _surface;
	VkQueue _graphicsQueue;
	VkQueue _presentQueue;
	VkQueue _computeQueue;

	VmaAllocator _allocator;

	const std::vector<const char*> _validationLayers = { "VK_LAYER_KHRONOS_validation" };
	const std::vector<const char*> _deviceExtensions = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME
	};
};

}

#endif
