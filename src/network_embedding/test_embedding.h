#include "storage/mgraph.h"
#include "solver/solver.h"

class Test_Embedding:public sae::Solver<int > {
public:
	Test_Embedding(sae::io::MappedGraph *graph);
	~Test_Embedding();
	int solve(int R, int T, int d, int w,int S, int TH, int Neg, double init_rate, std::map<sae::io::vid_t, sae::io::vid_t> mapToSae, std::string label_file,std::string output_file);
private:
};
