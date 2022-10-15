#include "bvh.cuh"

namespace nagi {

BVH::~BVH() {
	if (devTree) {
		cudaFree(devTree);
		checkCUDAError("cudaFree devTree failed.");
	}
	//if (devTreeTrigIdx) {
	//	cudaFree(devTreeTrigIdx);
	//	checkCUDAError("cudaFree devTree failed.");
	//}
}

void BVH::build() {
	std::chrono::steady_clock::time_point timer;
	timer = std::chrono::high_resolution_clock::now();
	std::cout << "Building BVH... ";

 	treeRoot = buildNode(0, MAX_TREE_DEPTH, scene.trigBuf.begin(), scene.trigBuf.end()-1, scene.bbox);

	//cudaMalloc((void**)&devTreeTrigIdx, sizeof(int) * trigIndices->size());
	//checkCUDAError("cudaMalloc devTreeTrigIdx failed.");

	//int i = 0;
	//for (auto it = trigIndices->begin(); it != trigIndices->end(); it++) {
	//	cudaMemcpy(devTreeTrigIdx + i, &it->first, sizeof(int), cudaMemcpyHostToDevice);
	//	i++;
	//}
	//checkCUDAError("cudaMemcpy devTreeTrigIdx failed.");

	cudaMalloc((void**)&devTree, sizeof(Node) * tree.size());
	checkCUDAError("cudaMalloc devTree failed.");

	cudaMemcpy(devTree, tree.data(), sizeof(Node) * tree.size(), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy devTree failed.");

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "Done. Time cost : " << runningTime << " seconds." << std::endl;
}

// reference: https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
int BVH::buildNode(
	int layer, int maxLayer, std::vector<Triangle>::iterator& trigStart, std::vector<Triangle>::iterator& trigEnd, Bound bbox) {
	int trigSize = trigEnd - trigStart + 1;
	if (trigSize == 0) return -1;
	if (trigSize > TERMINATE_NUM && 
		layer != maxLayer && 
		bbox.halfExtent.x > BVH_EPSILON && 
		bbox.halfExtent.y > BVH_EPSILON && 
		bbox.halfExtent.z > BVH_EPSILON) {

		Bound centerBox{};
		for (auto it = trigStart; it <= trigEnd; it++) {
			updateBoundingBox(it->bbox.center, &centerBox);
		}

		int axis;
		if (centerBox.halfExtent.x >= centerBox.halfExtent.y && centerBox.halfExtent.x >= centerBox.halfExtent.z) {
			axis = 0;
		}
		else if (centerBox.halfExtent.y >= centerBox.halfExtent.x && centerBox.halfExtent.y >= centerBox.halfExtent.z) {
			axis = 1;
		}
		else {
			axis = 2;
		}
		Bound b0, b1;
		std::vector<Triangle>::iterator middle;

		auto comparator = [&, axis](const Triangle& a, const Triangle& b) {
			if (axis == 0) return a.bbox.center.x < b.bbox.center.x;
			if (axis == 1) return a.bbox.center.y < b.bbox.center.y;
			return a.bbox.center.z < b.bbox.center.z;
		};

		std::sort(trigStart, trigEnd+1, comparator);

		auto area = [&, axis](const Bound& bbox) {
			if (axis == 0) return bbox.halfExtent.y * bbox.halfExtent.z;
			if (axis == 1) return bbox.halfExtent.x * bbox.halfExtent.z;
			if (axis == 2) return bbox.halfExtent.x * bbox.halfExtent.y;
		};

		if (trigSize > BUCKET_NUM) {
			Bucket buckets[BUCKET_NUM] = {};
			for (int i = 0; i < BUCKET_NUM; i++) {
				auto& bucket = buckets[i];
				if (i == 0) bucket.begin = trigStart;
				else bucket.begin = buckets[i - 1].end + 1;
				auto it = bucket.begin;

				if (i == BUCKET_NUM - 1) {
					bucket.end = trigEnd;
				}
				else {
					bucket.end = trigStart + trigSize / BUCKET_NUM * (i + 1);
				}
				while (it != bucket.end) {
					updateBoundingBox(it->bbox.min, &bucket.bbox);
					updateBoundingBox(it->bbox.max, &bucket.bbox);
					it++;
				}
				updateBoundingBox(it->bbox.min, &bucket.bbox);
				updateBoundingBox(it->bbox.max, &bucket.bbox);
			}

			float minRatio = FLT_MAX;
			for (int i = 0; i < BUCKET_NUM - 1; i++) {
				Bound tmp{}, tmp2{};
				for (int j = 0; j <= i; j++) {
					updateBoundingBox(buckets[j].bbox.min, &tmp);
					updateBoundingBox(buckets[j].bbox.max, &tmp);
				}
				for (int j = i + 1; j < BUCKET_NUM; j++) {
					updateBoundingBox(buckets[j].bbox.min, &tmp2);
					updateBoundingBox(buckets[j].bbox.max, &tmp2);
				}
				float area1 = area(tmp), area2 = area(tmp2);
				float ratio = fmaxf(area1, area2) / fminf(area1, area2);
				if (ratio < minRatio) {
					minRatio = ratio;
					b0 = tmp;
					b1 = tmp2;
					middle = buckets[i].end;
				}
			}
		}
		else {
			for (auto i = trigStart; i < trigEnd; i++) {
				float minRatio = FLT_MAX;
				Bound tmp{}, tmp2{};
				for (auto j = trigStart; j <= i; j++) {
					updateBoundingBox(j->bbox.min, &tmp);
					updateBoundingBox(j->bbox.max, &tmp);
				}
				for (auto j = i + 1; j <= trigEnd; j++) {
					updateBoundingBox(j->bbox.min, &tmp2);
					updateBoundingBox(j->bbox.max, &tmp2);
				}
				float area1 = area(tmp), area2 = area(tmp2);
				float ratio = fmaxf(area1, area2) / fminf(area1, area2);
				if (ratio < minRatio) {
					minRatio = ratio;
					b0 = tmp;
					b1 = tmp2;
					middle = i;
				}
			}
		}

		Node node{
			false,
			buildNode(layer + 1, maxLayer, trigStart, middle, b0),
			buildNode(layer + 1, maxLayer, middle + 1, trigEnd, b1),
			bbox.min,
			bbox.max,
			b0.min,
			b0.max,
			b1.min,
			b1.max
		};

		//generateBoundingBox(b0.min - BVH_EPSILON, b0.max + BVH_EPSILON, &b0);
		//generateBoundingBox(b1.min - BVH_EPSILON, b1.max + BVH_EPSILON, &b1);

		if (node.left == -1 && node.right == -1) {
			return -1;
		}

		if (node.left != -1 && node.right == -1) {
			return node.right;
		}

		if (node.left == -1 && node.right != -1) {
			return node.left;
		}

		tree.push_back(std::move(node));
		return tree.size() - 1;
	}
	else {
		// construct a leaf node
		Node node{
			true,
			trigStart - scene.trigBuf.begin(),
			trigEnd - scene.trigBuf.begin(),
			bbox.min,
			bbox.max
		};
		tree.push_back(std::move(node));
		//std::cout << layer << " " << trigSize << std::endl;
		return tree.size() - 1;
	}
}

}