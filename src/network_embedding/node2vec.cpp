#include "node2vec.h"
#include "word2vec.h"
#include <vector>
#include <stack>
#include <algorithm>
#include <iostream>
#include <initializer_list>
using namespace std;
using namespace sae::io;

Node2Vec::Node2Vec(MappedGraph *graph)
	:Solver(graph) {
}

Node2Vec::~Node2Vec() {
}

vector<vector<vid_t>> generate_2nd_paths(MappedGraph* graph,int R,int T)
{
    double p = 1,q=0.5;
    int n=graph->VertexCount();
    vector<vid_t> temp(n,0);
    vector<vector<vid_t>> paths;
    vector<vector<vid_t>> neighbor(n);
    vector<vector<vector<vid_t>>> alias_edges(n);
    vector<vector<vector<double>>> probs(n);
    auto viter = graph->Vertices(),viter2 = graph->Vertices();
    for(int i=0;i<n;i++)
    {
        temp[i] = i;
        viter->MoveTo(i);
        vector<vector<vid_t>> i_neighbor;
        vector<vector<double>> i_neighbor_probs;
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
        {
            neighbor[i].push_back(eiter->TargetId());
            vid_t u = eiter->TargetId();
            viter2->MoveTo(u);
            vector<vid_t> u_neighbor;
            vector<double> u_neighbor_probs;
            for(auto eiter2 = viter2->OutEdges(); eiter2->Alive(); eiter2->Next())
            {
                vid_t w = eiter2->TargetId();
                u_neighbor.push_back(w);
                if(w==i)
                    u_neighbor_probs.push_back(1.0/p);
                else if(find(neighbor[i].begin(),neighbor[i].end(),w)!=neighbor[i].end())
                    u_neighbor_probs.push_back(1.0);
                else
                    u_neighbor_probs.push_back(1.0/q);
            }
            double sum = accumulate(u_neighbor_probs.begin(),u_neighbor_probs.end(),0);
            for(int j=0;j<u_neighbor_probs.size();j++)
                u_neighbor_probs[j]/=sum;
            u_neighbor = Word2Vec::alias_setup(u_neighbor_probs);
            i_neighbor.push_back(u_neighbor);
            i_neighbor_probs.push_back(u_neighbor_probs);
        }
        alias_edges[i] = i_neighbor;
        probs[i] = i_neighbor_probs;
    }

    for (int i=0;i<R;i++)
    {
        vector<vid_t> rand_seq = temp;
        for (int j =n-1;j>=0;j--)
        {
                int x = rand() % (n - 1);
                unsigned tem =rand_seq[j];
                rand_seq[j]=rand_seq[x];
                rand_seq[x]=tem;
        }
        for(int j=0;j<n;j++)
        {
            vector<vid_t> path(T);
            vid_t s=rand_seq[j];
            path[0]=s;
            int now = rand()%neighbor[s].size();
            vid_t now_in = neighbor[s][now];
            path[1] = now_in;
            for(int j=2;j<T;j++)
            {
                int m=neighbor[now_in].size();
                if(m==0) break;
                vid_t  rand_num=Word2Vec::alias_draw(alias_edges[s][now],probs[s][now]) ;
                vid_t v=neighbor[now_in][rand_num];
                s=now_in;
                now  = rand_num;
                now_in = v;
                path[j]=v;
            }
            paths.push_back(path);
        }
    }
	return paths;
}


std::vector<std::vector<double> >  Node2Vec::solve(int R, int T, int d, int w) {

    /*
    vector<double> p;
    p.push_back(0.3);p.push_back(0.3);p.push_back(0.4);
    vector<vid_t> J = Word2Vec::alias_setup(p);
    vector<int> cnt(p.size(),0);
    int test_num = 10000;
    for(int i=0;i<test_num;i++)
    {
        vid_t s = Word2Vec::alias_draw(J,p);
        cnt[s] ++;
    }
    for(int i=0;i<p.size();i++)
        printf("now pro:%.4f\tfre:%.4f\n",p[i],cnt[i]*1.0/test_num);
    */

    vector<vector<vid_t>> sents = generate_2nd_paths(graph,R,T);
    Word2Vec wv(graph);
    vector<vector<double> >  ans = wv.solve(sents,d,w,10,10,0.025);
    return ans;
}
