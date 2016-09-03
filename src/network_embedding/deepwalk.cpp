#include "deepwalk.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include "../lib/word2vec/word2vec.h"
#include <initializer_list>
using namespace std;
using namespace sae::io;

Deep_Walk::Deep_Walk(MappedGraph *graph)
	:Solver(graph) {
}

Deep_Walk::~Deep_Walk() {
}

vector<vector<vid_t>> generate_paths(MappedGraph* graph,int R,int T)
{
    int n=graph->VertexCount();
    vector<vid_t> temp(n,0);
    vector<vector<vid_t>> paths;
    vector<vector<vid_t>> outdegree(n);
    auto viter = graph->Vertices();
    for(int i=0;i<n;i++)
    {
        temp[i] = i;
        viter->MoveTo(i);
        for(auto eiter = viter->OutEdges(); eiter->Alive(); eiter->Next())
                    outdegree[i].push_back(eiter->TargetId());
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
            int s=rand_seq[j];
            path[0]=s;
            for(int j=1;j<T;j++)
            {
                int m=outdegree[s].size();
                if(m==0) break;
                int  rand_num=rand() % m ;
                int v=outdegree[s][rand_num];
                s=v;
                path[j]=s;
            }
            paths.push_back(path);
        }
    }
	return paths;
}


std::vector<std::vector<double> >  Deep_Walk::solve(int R, int T, int d, int w) {

    Word2Vec<std::string> model(d,w);
	using Sentence = Word2Vec<std::string>::Sentence;
	using SentenceP = Word2Vec<std::string>::SentenceP;
    std::vector<SentenceP> sentences;
    SentenceP sentence(new Sentence);
	model.sample_ = 0;
	int n_workers = 4;

    vector<vector<vid_t>> sents = generate_paths(graph,R,T);
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
