#include "vk_shader.hpp"

namespace nagi {

Shader::Shader(
	Device& device,
	const std::string& filePath,
	VkShaderStageFlagBits stage)
	: _device{ device } {

	auto codes = readSpv(filePath);
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = codes.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(codes.data());
	if (vkCreateShaderModule(_device.device(), &createInfo, nullptr, &_module) != VK_SUCCESS) {
		throw std::runtime_error("Error: Failed to create vertex shader module.");
	}

	_stage = stage;

	_pCompiler = std::make_unique<spirv_cross::Compiler>(reinterpret_cast<const uint32_t*>(codes.data()), codes.size() / sizeof(uint32_t));
}

Shader::~Shader() {
	vkDestroyShaderModule(device(), _module, nullptr);
}

VkPipelineShaderStageCreateInfo Shader::getCreateInfo() {
	VkPipelineShaderStageCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	createInfo.stage = _stage;
	createInfo.module = _module;
	createInfo.pName = "main";
	createInfo.flags = 0;
	createInfo.pNext = nullptr;
	createInfo.pSpecializationInfo = nullptr;
	return createInfo;
}

}