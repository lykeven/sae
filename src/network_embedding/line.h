#include "storage/mgraph.h"
#include "solver/solver.h"

class LINE:public sae::Solver<std::vector<std::vector<double> > > {
public:
	LINE(sae::io::MappedGraph *graph);
	~LINE();
	std::vector<std::vector<double> > solve(int T, int d);
  
private:
};
