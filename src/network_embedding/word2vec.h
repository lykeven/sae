#include "storage/mgraph.h"
#include "solver/solver.h"

class Word2Vec:public sae::Solver<std::vector<std::vector<double> > > {
public:
    Word2Vec(sae::io::MappedGraph *graph);
	~Word2Vec();
	std::vector<std::vector<double> > solve(std::vector<std::vector<sae::io::vid_t> >corpus, int D, int W, int I, int K, double alpha);

	static std::vector<sae::io::vid_t> alias_setup(std::vector<double> &prob);
	static sae::io::vid_t alias_draw(std::vector<sae::io::vid_t> &J,std::vector<double> &p);

private:
};
