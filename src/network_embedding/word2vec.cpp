#include "word2vec.h"
#include <vector>
#include <stack>
#include <cmath>
#include <ctime>
#include <random>
#include <algorithm>
#include <iostream>
#include <pthread.h>

#define SIGMOID_BOUND 6
#define SIGMOID_TABLE_SIZE 2000
#define NEG_SAMPLING_POWER 0.75

using namespace std;
using namespace sae::io;

int dimension = 100, windows = 5, num_neg = 10, num_thread = 10;
int num_nodes, num_sentences;
double fixed_alpha;

vector<vector<vid_t>> all_corpus;
vector<vector<double>> emb_nodes;
vector<vector<double>> emb_nodes_sample;

vector<double> sigm_table;
vector<double> nodes_prob;
vector<vid_t> nodes_table;

struct argsformat
{
    int id;
};

Word2Vec::Word2Vec(MappedGraph *graph)
    : Solver(graph)
{
}

Word2Vec::~Word2Vec()
{
}

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

void *TrainThread(void *args)
{
    int id = ((argsformat *)args)->id;
    int start_sentence = (double)id / num_thread * num_sentences;
    int end_sentence = ((double)id + 1.0) / num_thread * num_sentences;
    for (int i = start_sentence; i < end_sentence; i++)
    {
        for (int j = 0; j < all_corpus[i].size(); j++)
        {
            vector<double> C(dimension, 0);
            vector<double> E(dimension, 0);
            vid_t word = all_corpus[i][j];
            //compute C
            for (int p = j - windows; p <= j + windows; p++)
            {
                vid_t pos = all_corpus[i][p];
                if (p < 0 || p >= all_corpus[i].size() || p == j)
                    continue;
                for (int d = 0; d < dimension; d++)
                    C[d] += emb_nodes[pos][d];
            }
            //negative sampling
            int label = 0;
            vid_t target = word;
            for (int k = 0; k < num_neg; k++)
            {
                if (k == 0)
                {
                    label = 1;
                    target = word;
                }
                else
                {
                    label = 0;
                    target = Word2Vec::alias_draw(nodes_table, nodes_prob);
                }
                //compute gradient for SGD
                double f = 0, g = 0;
                for (int d = 0; d < dimension; d++)
                    f += emb_nodes_sample[target][d] * C[d];
                if (f > SIGMOID_BOUND)
                    g = fixed_alpha * (label - 1);
                else if (f < -SIGMOID_BOUND)
                    g = fixed_alpha * (label - 0);
                else
                    g = fixed_alpha * (label - sigm_table[floor((f + SIGMOID_BOUND) * SIGMOID_TABLE_SIZE / SIGMOID_BOUND / 2)]);

                //accumulate error for word vector
                for (int d = 0; d < dimension; d++)
                    E[d] += g * emb_nodes_sample[target][d];
                //update map matrix
                for (int d = 0; d < dimension; d++)
                    emb_nodes_sample[target][d] += g * C[d];
            }
            //update each context for vector of w
            for (int p = j - windows; p <= j + windows; p++)
            {
                vid_t pos = all_corpus[i][p];
                if (p < 0 || p >= all_corpus[i].size() || p == j)
                    continue;
                for (int d = 0; d < dimension; d++)
                    emb_nodes[pos][d] += E[d];
            }
        }
    }
}

vector<vector<double>> Word2Vec::solve(vector<vector<vid_t>> corpus, int D, int W, int I, int K, double init_alpha)
{
    //initialize basic parameters
    all_corpus = corpus;
    dimension = D;
    windows = W;
    num_neg = K;
    num_thread = I;
    fixed_alpha = init_alpha;
    num_sentences = corpus.size();
    num_nodes = graph->VertexCount();
    srand(time(NULL));

    //initialize node embedding vector
    emb_nodes = vector<vector<double>>(num_nodes, vector<double>(dimension, 0));
    emb_nodes_sample = vector<vector<double>>(num_nodes, vector<double>(dimension, 0));
    for (int i = 0; i < num_nodes; i++)
        for (int j = 0; j < dimension; j++)
            emb_nodes[i][j] = (rand() / (RAND_MAX + 0.0) - 0.5) / dimension;

    //set up sample table with alias sampling
    nodes_prob = vector<double>(num_nodes, 0);
    for (int i = 0; i < all_corpus.size(); i++)
        for (int j = 0; j < all_corpus[i].size(); j++)
            nodes_prob[all_corpus[i][j]] += 1;
    for (int i = 0; i < num_nodes; i++)
        nodes_prob[i] = pow(nodes_prob[i], NEG_SAMPLING_POWER);
    double sum = accumulate(nodes_prob.begin(), nodes_prob.end(), 0);
    for (int i = 0; i < num_nodes; i++)
        nodes_prob[i] /= sum;
    nodes_table = Word2Vec::alias_setup(nodes_prob);

    //set up exp table
    sigm_table = vector<double>(SIGMOID_TABLE_SIZE, 0);
    for (int i = 0; i < SIGMOID_TABLE_SIZE; i++)
    {
        sigm_table[i] = exp((i * 1.0 / SIGMOID_TABLE_SIZE * 2 - 1) * SIGMOID_BOUND);
        sigm_table[i] = sigm_table[i] / (sigm_table[i] + 1);
    }

    //set up multi-thread
    pthread_t *pt = (pthread_t *)malloc(num_thread * sizeof(pthread_t));
    argsformat *args = (argsformat *)calloc(num_thread, sizeof(argsformat));
    if (pt == NULL)
    {
        fprintf(stderr, "Memory allocation failed.");
        exit(0);
    }
    for (int i = 0; i < num_thread; i++)
    {
        args[i].id = i;
        pthread_create(&pt[i], NULL, TrainThread, (void *)&args[i]);
    }

    for (int i = 0; i < num_thread; i++)
        pthread_join(pt[i], NULL);

    return emb_nodes;
}
