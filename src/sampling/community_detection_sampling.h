#include "storage/mgraph.h"
#include "solver/solver.h"

class Community_detection_sampling:public sae::Solver<std::pair<std::vector<sae::io::vid_t>,double>> {
public:
    Community_detection_sampling(sae::io::MappedGraph *graph);
    ~Community_detection_sampling();
    std::pair<std::vector<sae::io::vid_t>,double> solve(double p,int k);
private:
};
