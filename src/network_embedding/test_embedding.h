#include "storage/mgraph.h"
#include "solver/solver.h"

class Test_Embedding:public sae::Solver<std::vector<std::pair<double,double>> > {
public:
	Test_Embedding(sae::io::MappedGraph *graph);
	~Test_Embedding();
	std::vector<std::pair<double,double>> solve(int R, int T, int d, int w,int S, std::map<sae::io::vid_t, sae::io::vid_t> mapToSae, std::string label_file,std::string output_file);
private:
};