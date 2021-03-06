#include "deepwalk.h"
#include "word2vec.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <initializer_list>
using namespace std;
using namespace sae::io;

Deep_Walk::Deep_Walk(MappedGraph *graph)
    : Solver(graph)
{
}

Deep_Walk::~Deep_Walk()
{
}

vector<vector<vid_t>> generate_paths(MappedGraph *graph, int R, int T)
{
    int n = graph->VertexCount();
    vector<vid_t> temp(n, 0);
    vector<vector<vid_t>> paths;
    vector<vector<int>> outdegree(n);
    auto viter = graph->Vertices();
    for (int i = 0; i < n; i++)
    {
        temp[i] = i;
        viter->MoveTo(i);
        for (auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
            outdegree[i].push_back(eiter->TargetId());
    }
    for (int i = 0; i < R; i++)
    {
        vector<vid_t> rand_seq = temp;
        for (int j = n - 1; j >= 0; j--)
        {
            int x = rand() % (n - 1);
            unsigned tem = rand_seq[j];
            rand_seq[j] = rand_seq[x];
            rand_seq[x] = tem;
        }
        for (int j = 0; j < n; j++)
        {
            vector<vid_t> path(T);
            vid_t s = rand_seq[j];
            path[0] = s;
            for (int j = 1; j < T; j++)
            {
                int m = outdegree[s].size();
                if (m == 0)
                    break;
                int rand_num = rand() % m;
                vid_t v = outdegree[s][rand_num];
                s = v;
                path[j] = s;
            }
            paths.push_back(path);
        }
    }
    return paths;
}

std::vector<std::vector<double>> Deep_Walk::solve(int R, int T, int d, int w, int Th, int Neg, double init_rate)
{

    int n = graph->VertexCount();
    printf("Number of nodes: %d\n", n);
    printf("Number of walks: %d\n", n * R);
    printf("Data size (walks*length): %d\n", n * R * T);
    printf("Deepwalk walking...\n");
    vector<vector<vid_t>> sents = generate_paths(graph, R, T);
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
    printf("Walking completed!\n");
    Word2Vec wv(graph);
    printf("Training...\n");
    vector<vector<double>> ans = wv.solve(sents, d, w, Th, Neg, init_rate);
    printf("Training completed!\n");
    return ans;
}
