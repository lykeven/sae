#include "storage/mgraph.h"
#include "solver/solver.h"

class Word2Vec:public sae::Solver<std::vector<std::vector<double> > > {
public:
    Word2Vec(sae::io::MappedGraph *graph);
	~Word2Vec();
	std::vector<std::vector<double> > solve(std::vector<std::vector<sae::io::vid_t> > corpus, int D, int W, int K, int I, double alpha);

	int D;
	int W;
	int K;
	int I;
	double alpha;
	int V;
	int M;
	std::vector<std::vector<sae::io::vid_t>> corpus;
	std::vector<std::vector<double>> Vec;
	std::vector<std::vector<double>> R;

	std::vector<double> exp_table;
	std::vector<double> sam_pro;
	std::vector<sae::io::vid_t> sam_table;

	static std::vector<sae::io::vid_t> alias_setup(std::vector<double> &prob);
	static sae::io::vid_t alias_draw(std::vector<sae::io::vid_t> J,std::vector<double> p);

	static void* TrainThread(void* id);

private:
};
