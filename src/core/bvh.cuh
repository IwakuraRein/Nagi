#ifndef BVH_CUH
#define BVH_CUH

#include "common.cuh"
#include "intersection.cuh"

#define BVH_EPSILON 1e-6f
#define BUCKET_NUM 16

namespace nagi {

class BVH {
public:
	struct OriginalNode {
		bool leaf;
		bool leftNode;
		int parent, left, right; 
		Bound bbox;
	};
	struct Node {
		int trigIdx;
		int missLink;
		glm::vec3 min, max;
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
	int BVH::buildNode(std::vector<Triangle>::iterator& trigStart, std::vector<Triangle>::iterator& trigEnd, Bound bbox, bool leftNode);
	void convertTreeNode(const OriginalNode& orig);
	
	Scene& scene;
	std::vector<OriginalNode> originalTree;
	std::vector<Node> tree;
	Node* devTree{ nullptr };
	//int leafTrigs{ 0 };
	//std::shared_ptr<std::list<std::pair<int, Triangle*>>> trigIndices;
	//int* devTreeTrigIdx{ nullptr };
};

}

#endif // !BVH_CUH
