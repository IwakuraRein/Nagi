#ifndef DESCRIPTOR_HPP
#define DESCRIPTOR_HPP

#include "vk_device.hpp"

#include <map>
#include <unordered_map>

namespace nagi {

using Bindings = std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding>;

class DescriptorSetLayout {
public:
	class Builder {
	public:
		Builder(Device& _device) : _device{ _device } {}

		Builder& addBinding(
			uint32_t binding,
			VkDescriptorType descriptorType,
			VkShaderStageFlags stageFlags,
			uint32_t count = 1);
		std::unique_ptr<DescriptorSetLayout> build() const;
		Bindings bindings{};

	private:
		Device& _device;

	};

	DescriptorSetLayout(
		Device& _device, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings);
	~DescriptorSetLayout();
	DescriptorSetLayout(const DescriptorSetLayout&&) = delete;
	DescriptorSetLayout& operator=(const DescriptorSetLayout&&) = delete;

	VkDescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }

	std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings;

	size_t bindingCount() const { return bindings.size(); }

private:
	Device& _device;
	VkDescriptorSetLayout descriptorSetLayout;

	friend class DescriptorWriter;
};

class DescriptorPool {
public:
	class Builder {
	public:
		Builder(Device& device) : _device{ device } {}

		Builder& addPoolSize(VkDescriptorType descriptorType, uint32_t count);
		Builder& setPoolFlags(VkDescriptorPoolCreateFlags flags);
		Builder& setMaxSets(uint32_t count);
		std::unique_ptr<DescriptorPool> build() const;

	private:
		Device& _device;
		std::vector<VkDescriptorPoolSize> poolSizes{};
		uint32_t maxSets = 1000;
		VkDescriptorPoolCreateFlags poolFlags = 0;
	};

	DescriptorPool(
		Device& _device,
		uint32_t maxSets,
		VkDescriptorPoolCreateFlags poolFlags,
		const std::vector<VkDescriptorPoolSize>& poolSizes);
	~DescriptorPool();
	DescriptorPool(const DescriptorPool&) = delete;
	DescriptorPool& operator=(const DescriptorPool&) = delete;

	bool allocateDescriptorSet(
		const VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet& descriptor) const;

	void freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const;

	void resetPool();

	VkDescriptorPool getPool() const { return _descriptorPool; }

private:
	Device& _device;
	VkDescriptorPool _descriptorPool;

	friend class DescriptorWriter;
};

class DescriptorWriter {
public:
	DescriptorWriter(DescriptorSetLayout& _setLayout, DescriptorPool& _pool);

	DescriptorWriter& writeBuffer(uint32_t binding, VkDescriptorBufferInfo bufferInfo);
	DescriptorWriter& writeImage(uint32_t binding, VkDescriptorImageInfo imageInfo);

	bool build(VkDescriptorSet& set);
	void overwrite(VkDescriptorSet& set);
	void clear();
	size_t imageCount() const { return _imageCount; }
	size_t uniformCount() const { return _uniformCount; }
	size_t storageCount() const { return _storageCount; }
	size_t bufferCount() const { return _storageCount + _uniformCount; }

private:
	DescriptorSetLayout& _setLayout;
	DescriptorPool& _pool;
	std::vector<VkWriteDescriptorSet> _writes;
	std::map<uint32_t, VkDescriptorBufferInfo> _bufferInfos;
	std::map<uint32_t, VkDescriptorImageInfo> _imageInfos;
	size_t _imageCount{ 0 }, _uniformCount{ 0 }, _storageCount{ 0 };
};

}

#endif