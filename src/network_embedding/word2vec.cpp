#include "word2vec.h"
#include <vector>
#include <stack>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <iostream>
#include <pthread.h>

#define MAX_EXP 6
#define EXP_SIZE 2000

using namespace std;
using namespace sae::io;

Word2Vec::Word2Vec(MappedGraph *graph)
    : Solver(graph)
{
}

Word2Vec::~Word2Vec()
{
}

struct argsformat
{
    int id;
    Word2Vec *self;
};

vector<vid_t> Word2Vec::alias_setup(vector<double> &prob)
{
    int cnt = prob.size();
    vector<vid_t> J(cnt, 0);
    vector<double> q(cnt, 1.0);
    stack<vid_t> smaller, larger;
    for (int i = 0; i < prob.size(); i++)
    {
        vid_t v = i;
        q[v] = cnt * prob[i];
        if (q[v] < 1.0)
            smaller.push(v);
        else
            larger.push(v);
    }
    while (smaller.size() > 0 && larger.size() > 0)
    {
        vid_t small = smaller.top(), large = larger.top();
        smaller.pop();
        larger.pop();
        J[small] = large;
        q[large] = q[large] + q[small] - 1.0;
        if (q[large] < 1.0)
            smaller.push(large);
        else
            larger.push(large);
    }
    prob = q;
    return J;
}

vid_t Word2Vec::alias_draw(vector<vid_t> &J, vector<double> &p)
{
    int cnt = J.size();
    vid_t kk = floor(rand() % cnt);
    double rand_p = 1.0 * rand() / RAND_MAX;
    if (rand_p < p[kk])
        return kk;
    else
        return J[kk];
}

void *Word2Vec::TrainThread(void *args)
{

    int id = ((argsformat *)args)->id;
    Word2Vec *self = ((argsformat *)args)->self;
    int start_sen = (double)id / self->I * self->M;
    int end_sen = ((double)id + 1.0) / self->I * self->M;
    for (int i = start_sen; i < end_sen; i++)
    {
        vector<vid_t> corpus = self->corpus[i];
        for (int j = 0; j < corpus.size(); j++)
        {
            vector<double> C(self->D, 0);
            vector<double> E(self->D, 0);
            int word = corpus[j];
            //compute C
            for (int p = j - self->W; p <= j + self->W; p++)
            {
                int pos = corpus[p];
                if (p < 0 || p >= corpus.size() || p == j)
                    continue;
                for (int d = 0; d < self->D; d++)
                    C[d] += self->Vec[pos][d];
            }

            //negative sampling
            int label = 0;
            int sam_word = word;
            for (int k = 0; k < self->K; k++)
            {
                if (k == 0)
                {
                    label = 1;
                    sam_word = word;
                }
                else
                {
                    label = 0;
                    sam_word = self->alias_draw(self->sam_table, self->sam_pro);
                }
                //compute gradient for SGD
                double f = 0;
                for (int d = 0; d < self->D; d++)
                    f += self->R[sam_word][d] * C[d];
                double g = 0;
                if (f > MAX_EXP)
                    g = self->alpha * (label - 1);
                else if (f < -MAX_EXP)
                    g = self->alpha * (label - 0);
                else
                    g = self->alpha * (label - self->exp_table[floor((f + MAX_EXP) * EXP_SIZE / MAX_EXP / 2)]);

                //accumulate error for word vector
                for (int d = 0; d < self->D; d++)
                    E[d] += g * self->R[sam_word][d];
                //update map matrix
                for (int d = 0; d < self->D; d++)
                    self->R[sam_word][d] += g * C[d];
            }

            //update each context for vector of w
            for (int p = j - self->W; p <= j + self->W; p++)
            {
                int pos = corpus[p];
                if (p < 0 || p >= corpus.size() || p == j)
                    continue;
                for (int d = 0; d < self->D; d++)
                    self->Vec[pos][d] += E[d];
            }
        }
    }
}

std::vector<std::vector<double>> Word2Vec::solve(vector<vector<vid_t>> corpus, int D, int W, int K, int I, double alpha)
{
    //initialize basic parameters
    this->corpus = corpus;
    this->D = D;
    this->W = W;
    this->K = K;
    this->I = I;
    this->alpha = alpha;
    this->M = corpus.size();
    this->V = graph->VertexCount();
    srand(time(NULL));

    //initialize node embedding vector
    vector<vector<double>> temp(this->V);
    vector<vector<double>> temp1(this->V);
    for (int i = 0; i < temp.size(); i++)
        for (int j = 0; j < this->D; j++)
        {
            temp[i].push_back((rand() / (RAND_MAX + 0.0) - 0.5) / this->D);
            temp1[i].push_back(0);
        }
    this->Vec = temp;
    this->R = temp1;

    //set up sample table with alias sampling
    vector<double> temp2(this->V, 0);
    this->sam_pro = temp2;
    for (int i = 0; i < this->corpus.size(); i++)
        for (int j = 0; j < this->corpus[i].size(); j++)
            this->sam_pro[this->corpus[i][j]] += 1;
    for (int i = 0; i < this->V; i++)
        this->sam_pro[i] = pow(this->sam_pro[i], 0.75);
    double sum = accumulate(this->sam_pro.begin(), this->sam_pro.end(), 0);
    for (int i = 0; i < this->V; i++)
        this->sam_pro[i] /= sum;
    this->sam_table = Word2Vec::alias_setup(this->sam_pro);

    //set up exp table
    vector<double> temp3(EXP_SIZE, 0);
    this->exp_table = temp3;
    for (int i = 0; i < EXP_SIZE; i++)
    {
        this->exp_table[i] = exp((i * 1.0 / EXP_SIZE * 2 - 1) * MAX_EXP);
        this->exp_table[i] = this->exp_table[i] / (this->exp_table[i] + 1);
    }

    //set up multi-thread
    pthread_t *pt = (pthread_t *)malloc(I * sizeof(pthread_t));
    argsformat *args = (argsformat *)calloc(I, sizeof(argsformat));
    if (pt == NULL)
    {
        fprintf(stderr, "Memory allocation failed.");
        exit(0);
    }
    for (int i = 0; i < I; i++)
    {
        args[i].id = i;
        args[i].self = this;
        pthread_create(&pt[i], NULL, Word2Vec::TrainThread, (void *)&args[i]);
    }
    for (int i = 0; i < I; i++)
        pthread_join(pt[i], NULL);

    return this->Vec;
}
