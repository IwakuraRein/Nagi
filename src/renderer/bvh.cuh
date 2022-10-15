#ifndef BVH_CUH
#define BVH_CUH

#include "common.cuh"
#include "intersection.cuh"

#define MAX_TREE_DEPTH 24
#define TERMINATE_NUM 1
#define BVH_EPSILON 1e-6f
#define BUCKET_NUM 16

namespace nagi {

class BVH {
public:
	struct Node {
		bool leaf;
		int left, right; 
		glm::vec3 min, max, leftMin, leftMax, rightMin, rightMax;
	};
	struct Bucket {
		Bound bbox;
		std::vector<Triangle>::iterator begin;
		std::vector<Triangle>::iterator end;
	};
	BVH(Scene& Scene) : scene { Scene } {}
	~BVH();
	BVH(const BVH&) = delete;
	void operator=(const BVH&) = delete;

	void build();
	int BVH::buildNode(
		int layer, int maxLayer, std::vector<Triangle>::iterator& trigStart, std::vector<Triangle>::iterator& trigEnd, Bound bbox);
	
	Scene& scene;
	std::vector<Node> tree;
	Node* devTree{ nullptr };
	int treeRoot;
	//int leafTrigs{ 0 };
	//std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigIndices;
	//int* devTreeTrigIdx{ nullptr };
};

}

#endif // !BVH_CUH
