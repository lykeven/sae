#include "storage/mgraph.h"
#include "solver/solver.h"

class Deep_Walk:public sae::Solver<std::vector<std::vector<double> > > {
public:
	Deep_Walk(sae::io::MappedGraph *graph);
	~Deep_Walk();
	std::vector<std::vector<double> > solve(int R, int T, int d, int w);
private:
};
