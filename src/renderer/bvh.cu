#include "bvh.cuh"

namespace nagi {

BVH::~BVH() {
	if (devTree) {
		cudaFree(devTree);
		checkCUDAError("cudaFree devTree failed.");
	}
	if (devTreeTrigIdx) {
		cudaFree(devTreeTrigIdx);
		checkCUDAError("cudaFree devTree failed.");
	}
}

void BVH::build() {
	std::chrono::steady_clock::time_point timer;
	timer = std::chrono::high_resolution_clock::now();
	std::cout << "Building BVH... ";

	trigIndices = std::make_shared<std::list<std::pair<int, Triangle*>>>();
	//assume we must reach max tree depth when triangles are larger than 16384
	double depthBottom = pow(16384.0 / (double)TERMINATE_NUM, 1.0 / (double)MAX_TREE_DEPTH);
	for (auto& obj : scene.objBuf) {
		int trigCount = obj.trigIdxEnd - obj.trigIdxStart + 1;
		auto initialIndices = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		initialIndices->resize(trigCount);
		int i = obj.trigIdxStart;
		for (auto it = initialIndices->begin(); it != initialIndices->end(); it++) {
			it->first = i;
			it->second = &scene.trigBuf[i];
			i++;
		}
		if (trigCount <= TERMINATE_NUM) {
			tree.push_back(Node{ trigCount, {}, {}, {}, (int)trigIndices->size(), obj.bbox });
			obj.treeRoot = tree.size() - 1;
			//obj.treeDepth = 0;
			trigIndices->splice(trigIndices->end(), *initialIndices);
		}
		else {
			int depth = std::min((int)(log((double)trigCount) / log(depthBottom)), MAX_TREE_DEPTH);
			obj.treeRoot = buildNode(0, depth, initialIndices, obj.bbox);
			//obj.treeDepth = depth;
		}
		//std::cout << obj.treeRoot <<" "<< trigCount << " " << trigIndices->size() << " "<< log((double)trigCount) / log(depthBottom) << std::endl;
		auto& node = tree[obj.treeRoot];
		//std::cout << "    "<< node.size << " " << node.trigIdxStart << std::endl;
	}

	cudaMalloc((void**)&devTreeTrigIdx, sizeof(int) * trigIndices->size());
	checkCUDAError("cudaMalloc devTreeTrigIdx failed.");

	int i = 0;
	for (auto it = trigIndices->begin(); it != trigIndices->end(); it++) {
		cudaMemcpy(devTreeTrigIdx + i, &it->first, sizeof(int), cudaMemcpyHostToDevice);
		i++;
	}
	checkCUDAError("cudaMemcpy devTreeTrigIdx failed.");

	cudaMalloc((void**)&devTree, sizeof(Node) * tree.size());
	checkCUDAError("cudaMalloc devTree failed.");

	cudaMemcpy(devTree, tree.data(), sizeof(Node) * tree.size(), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devTree failed.");

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "Done. Tirangles: " << trigIndices->size() << ". Nodes: " << tree.size() << ". Time cost : " << runningTime << " seconds." << std::endl;
}

int BVH::buildNode(
	int layer, int maxLayer, std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs, BoundingBox bbox) {
	if (trigs->size() == 0) return -1;
	if (trigs->size() > TERMINATE_NUM && layer != maxLayer && bbox.halfExtent.x > FLT_EPSILON && bbox.halfExtent.y > FLT_EPSILON) {

		//glm::vec3 eps{ FLT_EPSILON, FLT_EPSILON, FLT_EPSILON };
		glm::vec3 eps{ 1e-6f, 1e-6f, 1e-6f };
		glm::vec3 halfX = glm::vec3{ bbox.halfExtent.x, 0.f, 0.f };
		glm::vec3 halfY = glm::vec3{ 0.f, bbox.halfExtent.y, 0.f };
		glm::vec3 halfZ = glm::vec3{ 0.f, 0.f, bbox.halfExtent.z };

		// prepare children's bounding boxes

		BoundingBox b0{}; updateBoundingBox(bbox.min - eps, bbox.center + eps, &b0);
		BoundingBox b1{}; updateBoundingBox(b0.min + halfZ - eps, b0.max + halfZ + eps, &b1);
		BoundingBox b2{}; updateBoundingBox(b0.min + halfY - eps, b0.max + halfY + eps, &b2);
		BoundingBox b3{}; updateBoundingBox(b0.min + halfY + halfZ - eps, b0.max + halfY + halfZ + eps, &b3);

		BoundingBox b4{}; updateBoundingBox(b0.min + halfX - eps, b0.max + halfX + eps, &b4);
		BoundingBox b5{}; updateBoundingBox(b4.min + halfZ - eps, b4.max + halfZ + eps, &b5);
		BoundingBox b6{}; updateBoundingBox(b4.min + halfY - eps, b4.max + halfY + eps, &b6);
		BoundingBox b7{}; updateBoundingBox(b4.min + halfY + halfZ - eps, b4.max + halfY + halfZ + eps, &b7);

		// store children's triangles
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs0 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs1 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs2 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs3 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs4 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs5 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs6 = std::make_shared<std::list<std::pair<int, Triangle*>>>();
		std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigs7 = std::make_shared<std::list<std::pair<int, Triangle*>>>();

		// find triangles
		for (auto it = trigs->begin(); it != trigs->end(); it++) {
			Triangle t = scene.trigBuf[it->first];
			if (tirgBoxIntersect(t, b0)) {
				trigs0->push_back(*it);
			}
			if (tirgBoxIntersect(t, b1)) {
				trigs1->push_back(*it);
			}
			if (tirgBoxIntersect(t, b2)) {
				trigs2->push_back(*it);
			}
			if (tirgBoxIntersect(t, b3)) {
				trigs3->push_back(*it);
			}
			if (tirgBoxIntersect(t, b4)) {
				trigs4->push_back(*it);
			}
			if (tirgBoxIntersect(t, b5)) {
				trigs5->push_back(*it);
			}
			if (tirgBoxIntersect(t, b6)) {
				trigs6->push_back(*it);
			}
			if (tirgBoxIntersect(t, b7)) {
				trigs7->push_back(*it);
			}
		}

		trigs.reset();

		Node node{
			0,
			{0},
			{},
			{},
			-1,
			bbox
		};

		int child;
		child = buildNode(layer + 1, maxLayer, trigs0, b0);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b0.min;
			node.childrenMax[node.size] = b0.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs1, b1);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b1.min;
			node.childrenMax[node.size] = b1.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs2, b2);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b2.min;
			node.childrenMax[node.size] = b2.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs3, b3);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b3.min;
			node.childrenMax[node.size] = b3.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs4, b4);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b4.min;
			node.childrenMax[node.size] = b4.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs5, b5);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b5.min;
			node.childrenMax[node.size] = b5.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs6, b6);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b6.min;
			node.childrenMax[node.size] = b6.max;
			node.size++;
		}

		child = buildNode(layer + 1, maxLayer, trigs7, b7);
		if (child >= 0) {
			node.children[node.size] = child;
			node.childrenMin[node.size] = b7.min;
			node.childrenMax[node.size] = b7.max;
			node.size++;
		}

		if (node.size > 0) {
			tree.push_back(std::move(node));
			return tree.size() - 1;
		}
		if (node.size == 1) {
			return node.children[0];
		}
		else return -1;
	}
	else {
		// construct a leaf node
		Node node{
			trigs->size(),
			{0},
			{},
			{},
			trigIndices->size(),
			bbox
		};
		trigIndices->splice(trigIndices->end(), *trigs);
		tree.push_back(std::move(node));
		return tree.size() - 1;
	}
}

}