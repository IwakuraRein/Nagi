#include "bvh.cuh"

namespace nagi {

BVH::~BVH() {
	if (devTree) {
		cudaRun(cudaFree(devTree));
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

 	int root = buildNode(scene.trigBuf.begin(), scene.trigBuf.end()-1, scene.bbox, false);
	
	tree.reserve(originalTree.size());
	for (auto& orig : originalTree) {
		convertTreeNode(orig);
	}

	cudaRun(cudaMalloc((void**)&devTree, sizeof(Node) * tree.size()));

	cudaRun(cudaMemcpy(devTree, tree.data(), sizeof(Node) * tree.size(), cudaMemcpyHostToDevice));

	float runningTime = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - timer).count();
	std::cout << "Done. Time cost : " << runningTime << " seconds." << std::endl;
}

// reference: https://www.pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
int BVH::buildNode(
	std::vector<Triangle>::iterator& trigStart, std::vector<Triangle>::iterator& trigEnd, Bound bbox, bool leftNode) {
	int trigSize = trigEnd - trigStart + 1;
	if (trigSize > 1) {
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

		//generateBoundingBox(bbox.min - BVH_EPSILON, bbox.max + BVH_EPSILON, &bbox);
		OriginalNode node{
			false,
			leftNode,
			-1,
			-1,
			-1,
			bbox
		};

		originalTree.push_back(std::move(node));
		int idx = originalTree.size() - 1;
		int left = buildNode(trigStart, middle, b0, true);
		originalTree[idx].left = left;
		int right = buildNode(middle + 1, trigEnd, b1, false);
		originalTree[idx].right = right;

		originalTree[left].parent = idx;
		originalTree[right].parent = idx;
		return idx;
	}
	else {
		// construct a leaf node
		OriginalNode node{
			true,
			leftNode,
			-1,
			trigStart - scene.trigBuf.begin(),
			trigStart - scene.trigBuf.begin(),
			bbox
		};
		originalTree.push_back(std::move(node));
		return originalTree.size() - 1;
	}
}
void BVH::convertTreeNode(const OriginalNode& orig) {
	int missLink;
	if (!orig.leaf) {
		missLink = -1;
		if (orig.leftNode && orig.parent >= 0) {
			missLink = originalTree[orig.parent].right;
		}
		if (!orig.leftNode && orig.parent >= 0) {
			auto& node = originalTree[orig.parent];
			while (!node.leftNode) {
				if (node.parent >= 0) node = originalTree[node.parent];
				else break;
			}
			if (node.leftNode && node.parent >= 0) missLink = originalTree[node.parent].right;
		}
	}
	else missLink = tree.size()+1;
	Node node{
		orig.leaf? orig.left : -1,
		missLink,
		orig.bbox.min,
		orig.bbox.max
	};
	tree.push_back(std::move(node));
}

}