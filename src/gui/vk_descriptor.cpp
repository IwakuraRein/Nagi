#include "vk_descriptor.hpp"

namespace nagi {

// *************** Descriptor Set Layout Builder *********************

DescriptorSetLayout::Builder& DescriptorSetLayout::Builder::addBinding(
	uint32_t binding,
	VkDescriptorType descriptorType,
	VkShaderStageFlags stageFlags,
	uint32_t count) {
	assert(bindings.count(binding) == 0 and "Binding already in use");
	VkDescriptorSetLayoutBinding layoutBinding{};
	layoutBinding.binding = binding;
	layoutBinding.descriptorType = descriptorType;
	layoutBinding.descriptorCount = count;
	layoutBinding.stageFlags = stageFlags;
	bindings[binding] = layoutBinding;
	return *this;
}

std::unique_ptr<DescriptorSetLayout> DescriptorSetLayout::Builder::build() const {
	return std::make_unique<DescriptorSetLayout>(_device, bindings);
}

// *************** Descriptor Set Layout *********************

DescriptorSetLayout::DescriptorSetLayout(
	Device& device, std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings)
	: _device{ device }, bindings{ bindings } {
	std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings{};
	for (auto kv : bindings) {
		setLayoutBindings.push_back(kv.second);
	}

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
	descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
	descriptorSetLayoutInfo.pBindings = setLayoutBindings.data();

	if (vkCreateDescriptorSetLayout(
		_device.device(),
		&descriptorSetLayoutInfo,
		nullptr,
		&descriptorSetLayout) != VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor set layout!");
	}
}

DescriptorSetLayout::~DescriptorSetLayout() {
	vkDestroyDescriptorSetLayout(_device.device(), descriptorSetLayout, nullptr);
}

// *************** Descriptor Pool Builder *********************

DescriptorPool::Builder& DescriptorPool::Builder::addPoolSize(
	VkDescriptorType descriptorType, uint32_t count) {
	poolSizes.push_back({ descriptorType, count });
	return *this;
}

DescriptorPool::Builder& DescriptorPool::Builder::setPoolFlags(
	VkDescriptorPoolCreateFlags flags) {
	poolFlags = flags;
	return *this;
}
DescriptorPool::Builder& DescriptorPool::Builder::setMaxSets(uint32_t count) {
	maxSets = count;
	return *this;
}

std::unique_ptr<DescriptorPool> DescriptorPool::Builder::build() const {
	return std::make_unique<DescriptorPool>(_device, maxSets, poolFlags, poolSizes);
}

// *************** Descriptor Pool *********************

DescriptorPool::DescriptorPool(
	Device& device,
	uint32_t maxSets,
	VkDescriptorPoolCreateFlags poolFlags,
	const std::vector<VkDescriptorPoolSize>& poolSizes)
	: _device{ device } {
	VkDescriptorPoolCreateInfo descriptorPoolInfo{};
	descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	descriptorPoolInfo.pPoolSizes = poolSizes.data();
	descriptorPoolInfo.maxSets = maxSets;
	descriptorPoolInfo.flags = poolFlags;

	if (vkCreateDescriptorPool(_device.device(), &descriptorPoolInfo, nullptr, &_descriptorPool) !=
		VK_SUCCESS) {
		throw std::runtime_error("failed to create descriptor pool!");
	}
}

DescriptorPool::~DescriptorPool() {
	vkDestroyDescriptorPool(_device.device(), _descriptorPool, nullptr);
}

bool DescriptorPool::allocateDescriptorSet(
	const VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet& descriptor) const {
	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.descriptorPool = _descriptorPool;
	allocInfo.pSetLayouts = &descriptorSetLayout;
	allocInfo.descriptorSetCount = 1;

	// Might want to create a "DescriptorPoolManager" class that handles this case, and builds
	// a new pool whenever an old pool fills up. But this is beyond our current scope
	if (vkAllocateDescriptorSets(_device.device(), &allocInfo, &descriptor) != VK_SUCCESS) {
		return false;
	}
	return true;
}

void DescriptorPool::freeDescriptors(std::vector<VkDescriptorSet>& descriptors) const {
	vkFreeDescriptorSets(
		_device.device(),
		_descriptorPool,
		static_cast<uint32_t>(descriptors.size()),
		descriptors.data());
}

void DescriptorPool::resetPool() {
	vkResetDescriptorPool(_device.device(), _descriptorPool, 0);
}

// *************** Descriptor Writer *********************

DescriptorWriter::DescriptorWriter(DescriptorSetLayout& _setLayout, DescriptorPool& _pool)
	: _setLayout{ _setLayout }, _pool{ _pool } {}

DescriptorWriter& DescriptorWriter::writeBuffer(
	uint32_t binding, VkDescriptorBufferInfo bufferInfo) {
	assert(_setLayout.bindings.count(binding) == 1 and "Layout does not contain specified binding");

	auto& bindingDescription = _setLayout.bindings[binding];

	assert(
		bindingDescription.descriptorCount == 1 and
		"Binding single descriptor info, but binding expects multiple");

	_bufferInfos[binding] = std::move(bufferInfo);

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.descriptorType = bindingDescription.descriptorType;
	write.dstBinding = binding;
	write.pBufferInfo = &_bufferInfos[binding];
	write.descriptorCount = 1;

	if (binding >= _writes.size()) {
		_writes.resize(binding + 1);
		if (bindingDescription.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
			_uniformCount++;
		else if (bindingDescription.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
			_storageCount++;
	}
	_writes[binding] = std::move(write);

	return *this;
}

DescriptorWriter& DescriptorWriter::writeImage(
	uint32_t binding, VkDescriptorImageInfo imageInfo) {
	assert(_setLayout.bindings.count(binding) == 1 and "Layout does not contain specified binding");

	auto& bindingDescription = _setLayout.bindings[binding];

	assert(
		bindingDescription.descriptorCount == 1 and
		"Binding single descriptor info, but binding expects multiple");

	_imageInfos[binding] = std::move(imageInfo);

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.descriptorType = bindingDescription.descriptorType;
	write.dstBinding = binding;
	write.pImageInfo = &_imageInfos[binding];
	write.descriptorCount = 1;

	if (binding >= _writes.size()) {
		_writes.resize(binding + 1);
		_imageCount++;
	}
	_writes[binding] = std::move(write);

	return *this;
}

bool DescriptorWriter::build(VkDescriptorSet& set) {
	bool success = _pool.allocateDescriptorSet(_setLayout.getDescriptorSetLayout(), set);
	if (!success) {
		return false;
	}
	overwrite(set);
	return true;
}

void DescriptorWriter::overwrite(VkDescriptorSet& set) {
	for (auto& write : _writes) {
		write.dstSet = set;
	}
	vkUpdateDescriptorSets(_pool._device.device(), _writes.size(), _writes.data(), 0, nullptr);
}

void DescriptorWriter::clear() {
	_writes.clear();
}

}