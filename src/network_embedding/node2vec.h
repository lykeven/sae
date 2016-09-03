#include "storage/mgraph.h"
#include "solver/solver.h"

class Node2Vec:public sae::Solver<std::vector<std::vector<double> > > {
public:
    Node2Vec(sae::io::MappedGraph *graph);
	~Node2Vec();
	std::vector<std::vector<double> > solve(int R, int T, int d, int w);
private:
};
