#ifndef BVH_CUH
#define BVH_CUH

#include "common.cuh"
#include "intersection.cuh"

#define MAX_TREE_DEPTH 10
#define TERMINATE_NUM 16

namespace nagi {

class BVH {
public:
	struct Node {
		// how many children has triangles (non leaf node)
		// how many triangles (leaf node)
		int size;
		int children[8];
		// store children's min and max. enough for ray-aabb intersection test
		glm::vec3 childrenMin[8];
		glm::vec3 childrenMax[8];
		int trigIdxStart; // if >=0, leaf node
		BoundingBox bbox;
	};
	BVH(Scene& Scene) : scene { Scene } {}
	~BVH();
	BVH(const BVH&) = delete;
	void operator=(const BVH&) = delete;

	void build();
	int BVH::buildNode(
		int layer, int maxLayer, std::shared_ptr<std::list<int>> trigs, BoundingBox bbox);
	
	Scene& scene;
	std::vector<Node> tree;
	int leafTrigs{ 0 };
	std::shared_ptr<std::list<int>> trigIndices;
	Node* devTree{ nullptr };
	int* devTreeTrigIdx{ nullptr };
};

}

#endif // !BVH_CUH
