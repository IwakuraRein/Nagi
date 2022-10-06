#ifndef SHADER_HPP
#define SHADER_HPP

#include "vk_device.hpp"
#include "vk_pipeline.hpp"

#include <spirv_cross/spirv_cross.hpp>

#include <fstream>

namespace nagi {
//a shader holds all of the shader related state that a pipeline needs to be built.

inline std::vector<char> readSpv(const std::string& filePath) {
	std::ifstream file{ filePath, std::ios::ate | std::ios::binary };

	if (!file.is_open()) {
		throw std::runtime_error("failed to open file: " + filePath);
	}

	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);

	file.close();
	return buffer;
}

class Shader {
public:
	Shader(
		Device& device,
		const std::string& filePath,
		VkShaderStageFlagBits stage);
	~Shader();
	Shader(const Shader&) = delete;
	Shader& operator=(const Shader&) = delete;
	Shader() = default;

	VkDevice device() const { return _device.device(); }
	VkShaderStageFlagBits stage() const { return _stage; }
	VkShaderModule module() const { return _module; }
	VkPipelineShaderStageCreateInfo getCreateInfo();

	friend class ShaderPass;
	friend class MaterialBase;
	friend class Material;

private:
	Device& _device;
	VkShaderModule _module;
	VkShaderStageFlagBits _stage;
	std::unique_ptr<spirv_cross::Compiler> _pCompiler;
};
}

#endif