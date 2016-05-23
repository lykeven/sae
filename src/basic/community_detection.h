#include "storage/mgraph.h"
#include "solver/solver.h"

class Community_detection:public sae::Solver<std::pair<std::vector<sae::io::vid_t>,double>> {
public:
    Community_detection(sae::io::MappedGraph *graph);
    ~Community_detection();
    double  compute_modularity(sae::io::MappedGraph *graph, std::vector<sae::io::eid_t> community,int k);
    std::pair<std::vector<sae::io::vid_t>,double>   run_Girvan_NewMan(sae::io::MappedGraph *graph,int k);
    std::pair<std::vector<sae::io::vid_t>,double>   run_label_propagation(sae::io::MappedGraph *graph);
    std::pair<std::vector<sae::io::vid_t>,double>   run_louvain_method(sae::io::MappedGraph *graph);
    std::pair<std::vector<sae::io::vid_t>,double>   run_k_community_core(sae::io::MappedGraph *graph,int k);
    std::pair<std::vector<sae::io::vid_t>,double> solve(int K,int sub_task);
private:
};
