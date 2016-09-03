#include "node2vec.h"
#include "deepwalk.h"
#include <vector>
#include <stack>
#include <algorithm>
#include <iostream>
#include "../lib/word2vec/word2vec.h"
#include <initializer_list>
using namespace std;
using namespace sae::io;

Node2Vec::Node2Vec(MappedGraph *graph)
	:Solver(graph) {
}

Node2Vec::~Node2Vec() {
}

vector<vid_t> alias_setup(vector<double> &prob)
{
    int cnt = prob.size();
    vector<vid_t> J (cnt,0);
    vector<double> q(cnt,1.0);
    stack <vid_t> smaller,larger;
    for(int i=0;i<prob.size();i++)
    {
        vid_t v = i;
        q[v] = cnt*prob[i];
        if(q[v]<1.0)
            smaller.push(v);
        else
            larger.push(v);
    }
    while(smaller.size()>0&&larger.size()>0)
    {
        vid_t small = smaller.top(),large =larger.top();
        smaller.pop();larger.pop();
        J[small] = large;
        q[large] = q[large] + q[small] -1.0;
        if(q[large]<1.0)
            smaller.push(large);
        else
            larger.push(large);
    }
    prob = q;
    return J;
}

vid_t alias_draw(vector<vid_t> J,vector<double> p)
{
    int cnt = J.size();
    vid_t kk = floor(random()%cnt);
    double rand_p = 1.0 * rand() / RAND_MAX;
    if (rand_p<p[kk])
        return kk;
    else
        return J[kk];
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
            u_neighbor = alias_setup(u_neighbor_probs);
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
                vid_t  rand_num=alias_draw(alias_edges[s][now],probs[s][now]) ;
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

    Word2Vec<std::string> model(d,w);
	using Sentence = Word2Vec<std::string>::Sentence;
	using SentenceP = Word2Vec<std::string>::SentenceP;
    std::vector<SentenceP> sentences;
    SentenceP sentence(new Sentence);
	model.sample_ = 0;
	int n_workers = 4;

    vector<vector<vid_t>> sents = generate_2nd_paths(graph,R,T);
    for(int i=0;i<sents.size();i++)
    {
        for(int j=0;j<sents[i].size();j++)
            sentence->tokens_.push_back(sae::serialization::cvt_to_string(sents[i][j]));
        sentence->words_.reserve(sentence->tokens_.size());
        sentences.push_back(move(sentence));
        sentence.reset(new Sentence);
    }
    model.build_vocab(sentences);
    printf("load vocab completed...\n");
    model.train(sentences, n_workers);
    printf("train completed...\n");
    vector<vector< double> > ans = model.wordVector();
    return ans;
}
