#include "node2vec.h"
#include "word2vec.h"
#include <vector>
#include <stack>
#include <map>
#include <algorithm>
#include <iostream>
#include <initializer_list>
using namespace std;
using namespace sae::io;

Node2Vec::Node2Vec(MappedGraph *graph)
    : Solver(graph)
{
}

Node2Vec::~Node2Vec()
{
}

vector<vector<vid_t>> generate_2nd_paths(MappedGraph *graph, int R, int T)
{
    double p = 1, q = 0.5;
    int n = graph->VertexCount();
    vector<vid_t> temp(n, 0);
    vector<vector<vid_t>> neighbor(n);
    vector<map<vid_t, bool>> is_neighbor(n);
    vector<vector<vector<vid_t>>> alias_edges(n);
    vector<vector<vector<double>>> alias_probs(n);
    auto viter = graph->Vertices(), viter2 = graph->Vertices();
    for (int i = 0; i < n; i++)
    {
        temp[i] = i;
        viter->MoveTo(i);
        for (auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
        {
            neighbor[i].push_back(eiter->TargetId());
            is_neighbor[i].insert(make_pair(eiter->TargetId(), true));
        }
    }
    for (int i = 0; i < n; i++)
    {
        vector<vector<vid_t>> i_neighbor;
        vector<vector<double>> i_neighbor_probs;
        for (int j = 0; j < neighbor[i].size(); j++)
        {
            vid_t u = neighbor[i][j];
            vector<vid_t> u_neighbor = neighbor[u];
            vector<double> u_neighbor_probs(neighbor[u].size(), 0);
            for (int k = 0; k < neighbor[u].size(); k++)
            {
                vid_t w = neighbor[u][k];
                if (w == i)
                    u_neighbor_probs[k] = 1.0 / p;
                else if (is_neighbor[i].find(w) != is_neighbor[i].end())
                    u_neighbor_probs[k] = 1.0;
                else
                    u_neighbor_probs[k] = 1.0 / q;
            }
            double sum = accumulate(u_neighbor_probs.begin(), u_neighbor_probs.end(), 0);
            for (int j = 0; j < u_neighbor_probs.size(); j++)
                u_neighbor_probs[j] /= sum;
            u_neighbor = Word2Vec::alias_setup(u_neighbor_probs);
            i_neighbor.push_back(u_neighbor);
            i_neighbor_probs.push_back(u_neighbor_probs);
        }
        alias_edges[i] = i_neighbor;
        alias_probs[i] = i_neighbor_probs;
    }

    vector<vector<vid_t>> paths;
    for (int i = 0; i < R; i++)
    {
        vector<vid_t> rand_seq = temp;
        for (int j = n - 1; j >= 0; j--)
            swap(rand_seq[j], rand_seq[rand() % (n - 1)]);
        for (int j = 0; j < n; j++)
        {
            vector<vid_t> path(T);
            vid_t pre_node_id = rand_seq[j];
            path[0] = pre_node_id;
            int cur_node = rand() % neighbor[pre_node_id].size();
            vid_t cur_node_id = neighbor[pre_node_id][cur_node];
            path[1] = cur_node_id;
            for (int j = 2; j < T; j++)
            {
                int m = neighbor[cur_node_id].size();
                if (m == 0)
                    break;
                vid_t next_node = Word2Vec::alias_draw(alias_edges[pre_node_id][cur_node], alias_probs[pre_node_id][cur_node]);
                vid_t next_node_id = neighbor[cur_node_id][next_node];
                pre_node_id = cur_node_id;
                cur_node = next_node;
                cur_node_id = next_node_id;
                path[j] = cur_node_id;
            }
            paths.push_back(path);
        }
    }
    return paths;
}

std::vector<std::vector<double>> Node2Vec::solve(int R, int T, int d, int w)
{

    int n = graph->VertexCount();
    printf("Number of nodes: %d\n", n);
    printf("Number of walks: %d\n", n * R);
    printf("Data size (walks*length): %d\n", n * R * T);
    printf("Node2vec walking...\n");
    vector<vector<vid_t>> sents = generate_2nd_paths(graph, R, T);
    /*
    string walk_file = "./output/walks.txt";
    FILE *walk = fopen(walk_file.c_str(), "wb");
    for(int i=0;i<sents.size();i++)
    {
        for(int j =0;j<sents[i].size();j++)
            fprintf(walk," %lld",sents[i][j]);
        fprintf(walk,"\n");
    }
    fclose(walk);
    printf("walks file saving completed!\n");
    */
    printf("walking completed!\n");
    Word2Vec wv(graph);
    printf("training...\n");
    vector<vector<double>> ans = wv.solve(sents, d, w, 10, 10, 0.025);
    printf("training completed!\n");
    return ans;
}
