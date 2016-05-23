#include "solver/solverForStreaming.h"
#include <vector>
#include <string>
#include <map>

typedef long long LL;
typedef std::uint64_t vid_t;
typedef std::uint64_t eid_t;

class resultMST
{
	public:
		double mstValue;
		vid_t n;
		eid_t m;
		std::vector<std::pair<std::pair<vid_t,vid_t>,double> > edge;
		resultMST()
		{
			n=m=mstValue=0;edge.clear();
		}
};

class lctNode
{
	public:
    	lctNode(int,double,lctNode*,lctNode*);
    	void Reverse();
    	void PushUp();
    	void PushDown();
    	lctNode *fa,*ls,*rs,*node1,*node2;
    	int num, maxEdge;
	double weight;
    	bool revMark;
};

class dynamicMinimumSpanningTree:public sae::SolverForStreaming<resultMST*>
{
	std::map<long long,int> p;
	std::map<long long,int>::iterator it1,it2;
	std::vector<long long> g;
	int n,m;
	std::vector<std::pair<std::pair<int,int>,double> > data_edge;
	void Zig(lctNode*);
	void Zag(lctNode*);
	void Splay(lctNode*);
	void Access(lctNode*);
	lctNode *findRoot(lctNode*);
	void moveToRoot(lctNode*);
	void Link(lctNode*,lctNode*);
	void Cut(lctNode*,lctNode*);
	int Query(lctNode*,lctNode*);
	int Insert(lctNode*,lctNode*,double);
	resultMST* _solve();
	public:
	dynamicMinimumSpanningTree(std::string file_path);
    	~dynamicMinimumSpanningTree();
    	resultMST* solve();  //return edge's two vertexs and its weight in MST
	resultMST* solve_raw(int);  //return edge's two vertexs and its weight in MST with raw data

};
