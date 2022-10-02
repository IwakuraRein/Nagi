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

	trigIndices = std::make_shared<std::list<int>>();
	auto initialIndices = std::make_shared<std::list<int>>();
	initialIndices->resize(scene.trigBuf.size());
	int i = 0;
	for (auto it = initialIndices->begin(); it != initialIndices->end(); it++) {
		*it = i;
		i++;
	}

	rootIdx = buildNode(0, initialIndices, scene.bbox);

	cudaMalloc((void**)&devTreeTrigIdx, sizeof(int) * trigIndices->size());
	checkCUDAError("cudaMalloc devTreeTrigIdx failed.");

	i = 0;
	for (auto it = trigIndices->begin(); it != trigIndices->end(); it++) {
		cudaMemcpy(devTreeTrigIdx + i, &(*it), sizeof(int), cudaMemcpyHostToDevice);
		i++;
	}
	checkCUDAError("cudaMemcpy devTreeTrigIdx failed.");

	cudaMalloc((void**)&devTree, sizeof(Node) * tree.size());
	checkCUDAError("cudaMalloc devTree failed.");

	cudaMemcpy(devTree, tree.data(), sizeof(Node) * tree.size(), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devTree failed.");

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "Done. Tirangles in BVH: " << tree.size() << ". Time cost : " << runningTime << " seconds." << std::endl;
}

int BVH::buildNode(
	int layer, std::shared_ptr<std::list<int>> trigs, BoundingBox bbox) {
	if (trigs->size() == 0) return -1;
	if (trigs->size() > TERMINATE_NUM && layer != MAX_TREE_DEPTH) {

		glm::vec3 eps{ FLT_EPSILON, FLT_EPSILON, FLT_EPSILON };
		eps = glm::max(eps, bbox.halfExtent * 0.01f);
		glm::vec3 halfX = glm::vec3{ bbox.halfExtent.x, 0.f, 0.f };
		glm::vec3 halfY = glm::vec3{ 0.f, bbox.halfExtent.y, 0.f };
		glm::vec3 halfZ = glm::vec3{ 0.f, 0.f, bbox.halfExtent.z };

		// prepare children's bounding boxes

		BoundingBox b0{}; updateBoundingBox(bbox.min, bbox.center, &b0);
		BoundingBox b1{}; updateBoundingBox(b0.min + halfZ, b0.max + halfZ, &b1);
		BoundingBox b2{}; updateBoundingBox(b0.min + halfY, b0.max + halfY, &b2);
		BoundingBox b3{}; updateBoundingBox(b0.min + halfY + halfZ, b0.max + halfY + halfZ, &b3);

		BoundingBox b4{}; updateBoundingBox(b0.min + halfX, b0.max + halfX, &b4);
		BoundingBox b5{}; updateBoundingBox(b4.min + halfZ, b4.max + halfZ, &b5);
		BoundingBox b6{}; updateBoundingBox(b4.min + halfY, b4.max + halfY, &b6);
		BoundingBox b7{}; updateBoundingBox(b4.min + halfY + halfZ, b4.max + halfY + halfZ, &b7);
		b0.min -= eps; b0.max += eps;
		b1.min -= eps; b1.max += eps;
		b2.min -= eps; b2.max += eps;
		b3.min -= eps; b3.max += eps;
		b4.min -= eps; b4.max += eps;
		b5.min -= eps; b5.max += eps;
		b6.min -= eps; b6.max += eps;
		b7.min -= eps; b7.max += eps;

		// store children's triangles
		std::shared_ptr<std::list<int>> trigs0{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs1{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs2{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs3{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs4{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs5{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs6{ new std::list<int> };
		std::shared_ptr<std::list<int>> trigs7{ new std::list<int> };

		// find triangles

		for (auto it = trigs->begin(); it != trigs->end(); it++) {
			Triangle t = scene.trigBuf[*it];
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
			-1,
			bbox
		};

		int child;
		child = buildNode(layer + 1, trigs0, b0);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs1, b1);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs2, b2);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs3, b3);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs4, b4);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs5, b5);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs6, b6);
		if (child >= 0) node.children[node.size++] = child;

		child = buildNode(layer + 1, trigs7, b7);
		if (child >= 0) node.children[node.size++] = child;

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
			trigIndices->size(),
			bbox
		};
		trigIndices->splice(trigIndices->end(), *trigs);
		tree.push_back(std::move(node));
		return tree.size() - 1;
	}
}

}